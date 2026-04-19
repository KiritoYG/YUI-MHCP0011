import stat
import shutil
import tempfile
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Request, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import asyncio
import time

from py.get_setting import EXT_DIR
from py.node_runner import node_mgr
from aiohttp import ClientSession

router = APIRouter(prefix="/api/extensions", tags=["extensions"])


class Extension(BaseModel):
    id: str
    name: str
    description: str = "无描述"
    version: str = "1.0.0"
    author: str = "未知"
    systemPrompt: str = ""
    repository: str = ""
    backupRepository: Optional[str] = ""
    category: str = ""
    transparent: bool = False
    width: int = 800
    height: int = 600
    enableVrmWindowSize: bool = False


class ExtensionsResponse(BaseModel):
    extensions: List[Extension]


class InstallResponse(BaseModel):
    ext_id: str
    status: str  # "installing", "success", "error"
    message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    status: str  # "installing", "success", "error", "unknown"
    detail: str
    progress: Optional[int] = None  # 0-100，可选
    timestamp: Optional[float] = None


# ==================== 工具函数 ====================

def _remove_readonly(func, path, exc_info):
    """Windows 只读文件处理回调"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def robust_rmtree(target: Path, preserve: Optional[set] = None):
    """安全删除目录，可选保留特定子目录"""
    target = Path(target)
    if not target.exists():
        return
    
    if preserve:
        temp_backup = {}
        for name in preserve:
            src = target / name
            if src.exists():
                tmp_dir = Path(tempfile.mkdtemp())
                dst = tmp_dir / name
                shutil.move(str(src), str(dst))
                temp_backup[name] = dst
        
        kwargs = {"onexc": _remove_readonly} if hasattr(shutil, "rmtree") and "onexc" in shutil.rmtree.__annotations__ else {"onerror": _remove_readonly}
        shutil.rmtree(target, **kwargs)
        
        target.mkdir(parents=True, exist_ok=True)
        for name, src in temp_backup.items():
            dst = target / name
            shutil.move(str(src), str(dst))
            shutil.rmtree(src.parent)
    else:
        kwargs = {"onexc": _remove_readonly} if hasattr(shutil, "rmtree") and "onexc" in shutil.rmtree.__annotations__ else {"onerror": _remove_readonly}
        shutil.rmtree(target, **kwargs)


def make_tree_writable(target: Path):
    """递归清除目录树的只读属性（Windows 专用）"""
    if os.name != 'nt':
        return
    for root, dirs, files in os.walk(target):
        for name in files:
            try:
                os.chmod(Path(root) / name, stat.S_IWRITE)
            except Exception:
                pass
        for name in dirs:
            try:
                os.chmod(Path(root) / name, stat.S_IWRITE)
            except Exception:
                pass


def find_root_dir(temp_path: Path) -> Path:
    """如果 zip 解压后只有 1 个一级目录且包含关键文件，则返回子目录"""
    entries = [p for p in temp_path.iterdir() if p.is_dir()]
    entry_files = ['index.html', 'index.js', 'package.json', 'manifest.json']
    
    if len(entries) == 1:
        subdir = entries[0]
        if any((subdir / f).exists() for f in entry_files):
            return subdir
    
    return temp_path


def compute_deps_hash(package_json_path: Path) -> Optional[str]:
    """计算依赖的指纹"""
    if not package_json_path.exists():
        return None
    
    try:
        with open(package_json_path, 'r', encoding='utf-8') as f:
            pkg = json.load(f)
        
        deps = {
            'dependencies': pkg.get('dependencies', {}),
            'devDependencies': pkg.get('devDependencies', {}),
            'engines': pkg.get('engines', {})
        }
        
        deps_str = json.dumps(deps, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(deps_str.encode()).hexdigest()[:16]
    except Exception:
        return None


def should_reuse_node_modules(old_pkg: Path, new_pkg: Path) -> bool:
    """判断是否可以复用 node_modules"""
    old_hash = compute_deps_hash(old_pkg)
    new_hash = compute_deps_hash(new_pkg)
    
    if old_hash is None or new_hash is None:
        return False
    
    return old_hash == new_hash


def github_url_to_zip(url: str) -> str:
    """将 GitHub/Gitee 仓库 URL 转换为 ZIP 下载链接"""
    url = url.strip().rstrip('/').removesuffix('.git')
    
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) < 2:
        raise ValueError(f"无效的仓库 URL: {url}")
    
    owner, repo = path_parts[0], path_parts[1]
    host = parsed.netloc.lower()
    
    if 'github.com' in host:
        return f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip"
    elif 'gitee.com' in host:
        return f"https://gitee.com/{owner}/{repo}/repository/archive/main.zip"
    else:
        return f"{url}/archive/refs/heads/main.zip"


# ==================== 安装任务管理 ====================

# 内存中的任务状态存储（生产环境建议用 Redis）
install_tasks: Dict[str, Dict[str, Any]] = {}


def update_task_status(ext_id: str, status: str, detail: str, progress: Optional[int] = None):
    """更新任务状态"""
    install_tasks[ext_id] = {
        "status": status,
        "detail": detail,
        "progress": progress,
        "timestamp": time.time()
    }


def get_ext_id_from_url(url: str) -> str:
    """从 URL 解析扩展 ID"""
    parsed = urlparse(url.strip().rstrip('/'))
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("无效的仓库 URL")
    return f"{path_parts[0]}_{path_parts[1]}"


class GitHubInstallRequest(BaseModel):
    url: str = Field(..., description="主仓库地址")
    backupUrl: Optional[str] = Field("", description="备用仓库地址")


# ==================== 核心安装逻辑 ====================

async def download_zip(url: str, dest: Path, timeout: float = 60.0) -> None:
    """异步下载 ZIP 文件"""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)


def _do_zip_install(zip_url: str, temp_dir: Path, target: Path, ext_id: str) -> None:
    """执行 ZIP 下载和解压安装"""
    
    old_pkg = target / "package.json"
    old_node_modules = target / "node_modules"
    can_reuse = False
    
    if old_pkg.exists() and old_node_modules.exists():
        try:
            zip_path = temp_dir / "new_repo.zip"
            asyncio.run(download_zip(zip_url, zip_path))
            
            temp_unpack = temp_dir / "preview"
            shutil.unpack_archive(zip_path, temp_unpack)
            new_root = find_root_dir(temp_unpack)
            new_pkg = new_root / "package.json"
            
            if should_reuse_node_modules(old_pkg, new_pkg):
                can_reuse = True
                update_task_status(ext_id, "installing", "依赖未变更，保留 node_modules", 30)
            else:
                update_task_status(ext_id, "installing", "依赖已变更，将重新安装", 30)
                
            shutil.rmtree(temp_unpack)
            
        except Exception as e:
            print(f"[{ext_id}] 无法比较依赖，将清理 node_modules: {e}")
            can_reuse = False
    else:
        zip_path = temp_dir / "new_repo.zip"
        asyncio.run(download_zip(zip_url, zip_path))
    
    if not can_reuse:
        robust_rmtree(target)
    else:
        robust_rmtree(target, preserve={'node_modules'})
    
    if not zip_path.exists():
        zip_path = temp_dir / "new_repo.zip"
        asyncio.run(download_zip(zip_url, zip_path))
    
    update_task_status(ext_id, "installing", "正在解压文件...", 50)
    
    unpack_dir = temp_dir / "unpacked"
    shutil.unpack_archive(zip_path, unpack_dir)
    
    new_root = find_root_dir(unpack_dir)
    make_tree_writable(new_root)
    
    if can_reuse:
        preserved_modules = target / "node_modules"
        temp_modules = temp_dir / "preserved_node_modules"
        
        if preserved_modules.exists():
            shutil.move(str(preserved_modules), str(temp_modules))
            robust_rmtree(target)
            shutil.move(str(new_root), str(target))
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_modules), str(preserved_modules))
        else:
            shutil.move(str(new_root), str(target))
    else:
        shutil.move(str(new_root), str(target))
    
    update_task_status(ext_id, "installing", "文件解压完成", 80)


def _run_bg_install(repo_url: str, ext_id: str, backup_url: str = ""):
    """后台安装任务"""
    update_task_status(ext_id, "installing", "正在准备安装...", 0)
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        target = Path(EXT_DIR) / ext_id
        target.parent.mkdir(parents=True, exist_ok=True)
        
        urls = []
        main = repo_url.strip().rstrip('/') if repo_url else ""
        backup = backup_url.strip().rstrip('/') if backup_url else ""
        
        update_task_status(ext_id, "installing", "检测网络环境...", 10)
        
        # 测试 GitHub 连通性
        try:
            import httpx
            with httpx.Client(timeout=3) as c:
                c.head("https://github.com")
            if main:
                urls.append(github_url_to_zip(main))
            if backup:
                urls.append(github_url_to_zip(backup))
        except Exception:
            if backup:
                urls.append(github_url_to_zip(backup))
            if main:
                urls.append(github_url_to_zip(main))
        
        if not urls:
            raise RuntimeError("没有可用的仓库地址")
        
        last_err = None
        for i, zip_url in enumerate(urls):
            update_task_status(ext_id, "installing", f"正在从源 {i+1}/{len(urls)} 下载...", 20)
            
            try:
                _do_zip_install(zip_url, temp_dir, target, ext_id)
                
                # 检查是否需要 npm install
                pkg_json = target / "package.json"
                node_modules = target / "node_modules"
                
                if pkg_json.exists() and not node_modules.exists():
                    update_task_status(ext_id, "installing", "正在安装 Node 依赖（可能需要几分钟）...", 85)
                    # 这里可以调用 npm install，如果需要的话
                    # 暂时跳过，由前端或下次启动时处理
                
                update_task_status(ext_id, "success", "安装完成", 100)
                return
                
            except Exception as e:
                last_err = e
                continue
        
        raise RuntimeError(f"所有源均下载失败: {last_err}")
        
    except Exception as e:
        update_task_status(ext_id, "error", str(e))
        target = Path(EXT_DIR) / ext_id
        if target.exists():
            robust_rmtree(target)
    finally:
        robust_rmtree(temp_dir)


def _run_zip_install(file_content: bytes, ext_id: str, filename: str = "upload.zip"):
    """处理本地上传 ZIP 的后台安装"""
    update_task_status(ext_id, "installing", "正在处理上传文件...", 0)
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        target = Path(EXT_DIR) / ext_id
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存上传的文件
        zip_path = temp_dir / filename
        with open(zip_path, "wb") as f:
            f.write(file_content)
        
        update_task_status(ext_id, "installing", "正在解压...", 30)
        
        # 解压并分析
        unpack_dir = temp_dir / "unpacked"
        shutil.unpack_archive(zip_path, unpack_dir)
        
        real_root = find_root_dir(unpack_dir)
        
        # 验证基本结构
        if not any((real_root / f).exists() for f in ['index.html', 'index.js', 'package.json']):
            raise ValueError("ZIP 内容不符合扩展格式（缺少 index.html/index.js/package.json）")
        
        update_task_status(ext_id, "installing", "正在安装...", 60)
        
        # 如果已存在，先删除
        if target.exists():
            robust_rmtree(target)
        
        target.mkdir(parents=True, exist_ok=True)
        make_tree_writable(real_root)
        
        for item in real_root.iterdir():
            shutil.move(str(item), str(target))
        
        update_task_status(ext_id, "success", "安装完成", 100)
        
    except Exception as e:
        update_task_status(ext_id, "error", str(e))
        target = Path(EXT_DIR) / ext_id
        if target.exists():
            robust_rmtree(target)
    finally:
        robust_rmtree(temp_dir)


# ==================== API 路由 ====================

@router.get("/list", response_model=ExtensionsResponse)
async def list_extensions():
    """获取所有可用的扩展列表"""
    try:
        extensions_dir = EXT_DIR
        
        if not os.path.exists(extensions_dir):
            os.makedirs(extensions_dir, exist_ok=True)
            return ExtensionsResponse(extensions=[])
        
        extensions = []
        for dir_name in os.listdir(extensions_dir):
            dir_path = os.path.join(extensions_dir, dir_name)
            if os.path.isdir(dir_path):
                ext_id = dir_name
                index_path = os.path.join(dir_path, "index.html")
                js_entry = os.path.join(dir_path, "index.js")
                
                if os.path.exists(index_path) or os.path.exists(js_entry):
                    package_path = os.path.join(dir_path, "package.json")
                    if os.path.exists(package_path):
                        try:
                            with open(package_path, 'r', encoding='utf-8') as f:
                                package_data = json.load(f)
                                
                            extensions.append(Extension(
                                id=ext_id,
                                name=package_data.get("name", ext_id),
                                description=package_data.get("description", "无描述"),
                                version=package_data.get("version", "1.0.0"),
                                author=package_data.get("author", "未知"),
                                systemPrompt=package_data.get("systemPrompt", ""),
                                repository=package_data.get("repository", ""),
                                backupRepository=package_data.get("backupRepository", ""),
                                category=package_data.get("category", ""),
                                transparent=package_data.get("transparent", False),
                                width=package_data.get("width", 800),
                                height=package_data.get("height", 600),
                                enableVrmWindowSize=package_data.get("enableVrmWindowSize", False)
                            ))
                        except json.JSONDecodeError:
                            extensions.append(Extension(id=ext_id, name=ext_id))
                    else:
                        extensions.append(Extension(id=ext_id, name=ext_id))
        
        return ExtensionsResponse(extensions=extensions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取扩展列表失败: {str(e)}")


@router.delete("/{ext_id}", status_code=204)
async def delete_extension(ext_id: str):
    """删除扩展"""
    target = Path(EXT_DIR) / ext_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="扩展不存在")
    try:
        robust_rmtree(target)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")


@router.post("/install-from-github", response_model=InstallResponse)
async def install_from_github(req: GitHubInstallRequest, background: BackgroundTasks):
    """从 GitHub/Gitee 安装扩展（后台任务+轮询）"""
    try:
        ext_id = get_ext_id_from_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    target = Path(EXT_DIR) / ext_id
    
    if target.exists():
        raise HTTPException(status_code=409, detail="扩展已存在，请使用更新接口")
    
    # 检查是否已有进行中的任务
    if ext_id in install_tasks and install_tasks[ext_id]["status"] == "installing":
        return InstallResponse(ext_id=ext_id, status="installing", message="安装任务已在进行中")
    
    background.add_task(_run_bg_install, req.url, ext_id, req.backupUrl or "")
    return InstallResponse(ext_id=ext_id, status="installing", message="后台安装任务已启动")


@router.get("/task-status/{ext_id}", response_model=TaskStatusResponse)
async def get_task_status(ext_id: str):
    """查询安装任务状态"""
    status = install_tasks.get(ext_id)
    if not status:
        # 检查是否已安装完成（任务可能被清理）
        target = Path(EXT_DIR) / ext_id
        if target.exists():
            return TaskStatusResponse(status="success", detail="已安装", timestamp=time.time())
        return TaskStatusResponse(status="unknown", detail="无此任务", timestamp=time.time())
    
    return TaskStatusResponse(**status)


@router.post("/upload-zip", response_model=InstallResponse)
async def upload_zip(file: UploadFile = File(...), background: BackgroundTasks = None):
    """上传本地 ZIP 安装扩展（改为后台任务+轮询模式）"""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="仅支持 zip 文件")
    
    ext_id = Path(file.filename).stem
    target = Path(EXT_DIR) / ext_id
    
    if target.exists():
        raise HTTPException(status_code=409, detail="扩展已存在")
    
    # 读取文件内容到内存
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="空文件")
    
    # 检查是否已有进行中的任务
    if ext_id in install_tasks and install_tasks[ext_id]["status"] == "installing":
        return InstallResponse(ext_id=ext_id, status="installing", message="安装任务已在进行中")
    
    # 启动后台任务
    background.add_task(_run_zip_install, content, ext_id, file.filename)
    
    return InstallResponse(ext_id=ext_id, status="installing", message="后台安装任务已启动")


@router.put("/{ext_id}/update")
async def update_extension(ext_id: str):
    """更新扩展（ZIP 方式，智能保留 node_modules）"""
    target = Path(EXT_DIR) / ext_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="扩展未安装")
    
    pkg_file = target / "package.json"
    if not pkg_file.exists():
        raise HTTPException(status_code=400, detail="缺少 package.json")
    
    try:
        meta = json.loads(pkg_file.read_text(encoding="utf-8"))
        repos = []
        if meta.get("repository"):
            repos.append(meta["repository"].strip().rstrip("/"))
        if meta.get("backupRepository"):
            repos.append(meta["backupRepository"].strip().rstrip("/"))
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析 package.json")
    
    if not repos:
        raise HTTPException(status_code=400, detail="缺少 repository 信息")
    
    try:
        with httpx.Client(timeout=3) as c:
            c.head("https://github.com")
        zip_urls = [github_url_to_zip(r) for r in repos]
    except Exception:
        zip_urls = [github_url_to_zip(r) for r in reversed(repos)]
    
    temp_dir = Path(tempfile.mkdtemp())
    last_err = None
    
    try:
        for zip_url in zip_urls:
            try:
                _do_zip_install(zip_url, temp_dir, target, ext_id)
                return {"status": "updated", "source": zip_url}
            except Exception as e:
                last_err = e
                continue
        
        raise HTTPException(status_code=500, detail=f"更新失败: {last_err}")
    finally:
        robust_rmtree(temp_dir)


# ==================== 远程插件列表 ====================

class RemotePluginItem(BaseModel):
    id: str
    name: str
    description: str
    author: str
    version: str
    category: str = "Unknown"
    repository: str
    backupRepository: Optional[str] = ""
    installed: bool = False


class RemotePluginList(BaseModel):
    plugins: List[RemotePluginItem]


@router.get("/remote-list", response_model=RemotePluginList)
async def remote_plugin_list():
    """获取远程插件列表"""
    github_raw = "https://raw.githubusercontent.com/super-agent-party/super-agent-party.github.io/main/plugins.json"
    gitee_raw = "https://gitee.com/super-agent-party/super-agent-party.github.io/raw/main/plugins.json"
    
    remote = None
    for url in (github_raw, gitee_raw):
        try:
            async with httpx.AsyncClient(timeout=10) as cli:
                r = await cli.get(url)
                r.raise_for_status()
                remote = r.json()
                break
        except Exception:
            if url == gitee_raw:
                raise HTTPException(
                    status_code=502,
                    detail="无法获取远程插件列表"
                )
            continue
    
    try:
        local_res = await list_extensions()
        installed_repos = {
            ext.repository.strip().rstrip("/").lower()
            for ext in local_res.extensions
            if ext.repository
        }
    except Exception:
        installed_repos = set()
    
    def _with_status(p: dict):
        repo = p.get("repository", "").strip().rstrip("/").lower()
        parse = urlparse(p.get("repository", ""))
        path_parts = parse.path.strip("/").split("/")
        ext_id = f"{path_parts[0]}_{path_parts[1]}" if len(path_parts) >= 2 else p.get("id", "")
        
        return RemotePluginItem(
            id=ext_id,
            name=p.get("name", "未命名"),
            description=p.get("description", ""),
            author=p.get("author", "未知"),
            version=p.get("version", "1.0.0"),
            category=p.get("category", "Unknown"),
            repository=p.get("repository", ""),
            backupRepository=p.get("backupRepository", ""),
            installed=repo in installed_repos,
        )
    
    return RemotePluginList(plugins=[_with_status(p) for p in remote])


# ==================== Node.js 支持 ====================

http_sess: ClientSession | None = None


@router.on_event("startup")
async def startup():
    global http_sess
    http_sess = ClientSession()


@router.on_event("shutdown")
async def shutdown():
    if http_sess:
        await http_sess.close()
    for ext_id in list(node_mgr.exts.keys()):
        await node_mgr.stop(ext_id)


@router.post("/{ext_id}/start-node")
async def start_node(ext_id: str):
    """启动 Node 扩展"""
    ext_dir = Path(EXT_DIR) / ext_id
    node_entry = ext_dir / "index.js"
    
    if not node_entry.exists():
        return {"mode": "static"}
    
    try:
        port = await node_mgr.start(ext_id)
        return {"mode": "node", "port": port}
    except Exception as e:
        node_modules = ext_dir / "node_modules"
        if not node_modules.exists():
            return {"mode": "error", "message": f"缺少依赖，请检查 node_modules: {e}"}
        return {"mode": "error", "message": str(e)}


@router.post("/{ext_id}/stop-node")
async def stop_node(ext_id: str):
    """停止 Node 扩展"""
    await node_mgr.stop(ext_id)
    return {"status": "stopped"}


@router.api_route("/{ext_id}/node/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(ext_id: str, path: str, request: Request):
    """代理 Node 扩展的 HTTP 请求"""
    if ext_id not in node_mgr.exts:
        raise HTTPException(404, "扩展未启动")
    
    port = node_mgr.exts[ext_id].port
    url = f"http://127.0.0.1:{port}/{path}"
    
    body = await request.body()
    async with http_sess.request(
        method=request.method,
        url=url,
        params=request.query_params,
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        data=body
    ) as resp:
        content = await resp.read()
        return Response(content, status_code=resp.status, headers=dict(resp.headers))