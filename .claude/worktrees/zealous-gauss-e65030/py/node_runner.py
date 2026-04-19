import os
import json, asyncio, socket
from pathlib import Path
from typing import Dict, Optional
from py.get_setting import EXT_DIR, IS_DOCKER

PORT_RANGE = (3100, 13999)

# 获取环境变量（由 Docker 或 Electron 注入）
ELECTRON_NODE = os.environ.get("ELECTRON_NODE_EXEC")
ELECTRON_NPM_CLI = os.environ.get("ELECTRON_NPM_CLI")

class NodeExtension:
    def __init__(self, ext_id: str):
        self.ext_id   = ext_id
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.port: Optional[int] = None
        self.root     = Path(EXT_DIR) / ext_id
        self.pkg      = json.loads((self.root / "package.json").read_text(encoding="utf-8"))

    def _get_exec_cmds(self):
        """智能生成 node 和 npm 的执行命令列表"""
        if IS_DOCKER or not ELECTRON_NODE:
            # Docker 或原生环境：直接使用系统全局的 node 和 npm
            npm_exe = "npm.cmd" if os.name == "nt" else "npm"
            return ["node"], [npm_exe]
        else:
            # Electron 桌面端环境：
            # Node 命令: electron.exe
            # NPM 命令: electron.exe /path/to/npm-cli.js
            return [ELECTRON_NODE], [ELECTRON_NODE, ELECTRON_NPM_CLI]

    def _get_env(self):
        """生成带 ELECTRON_RUN_AS_NODE 标记的环境变量"""
        env = os.environ.copy()
        if not IS_DOCKER and ELECTRON_NODE:
            env["ELECTRON_RUN_AS_NODE"] = "1"
        return env

    async def start(self) -> int:
        if self.proc and self.proc.returncode is None:
            return self.port

        pkg_file = self.root / "package.json"
        nm_folder = self.root / "node_modules"
        
        node_cmd, npm_cmd = self._get_exec_cmds()
        run_env = self._get_env()

        # 0. 快速判断：node_modules 存在且比 package.json 新
        if nm_folder.is_dir() and nm_folder.stat().st_mtime >= pkg_file.stat().st_mtime:
            print(f"[{self.ext_id}] node_modules 已存在，跳过 npm install")
        else:
            print(f"[{self.ext_id}] 首次/依赖变更，执行 npm install")
            # 1. 启动 npm install
            # 注意这里使用 *npm_cmd 解包列表
            proc = await asyncio.create_subprocess_exec(
                *npm_cmd, "install", "--production",
                cwd=self.root,
                env=run_env,  # 必须传入修改后的环境变量
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"npm install 失败:\n{stdout.decode('utf-8', errors='ignore')}")
            # 刷新时间戳
            nm_folder.touch(exist_ok=True)

        # 2. 选端口
        want = self.pkg.get("nodePort", 0)
        self.port = want if want else _free_port()

        # 3. 起进程
        self.proc = await asyncio.create_subprocess_exec(
            *node_cmd, "index.js", str(self.port),
            cwd=self.root,
            env=run_env, # 必须传入修改后的环境变量
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # 4. 等健康
        await _wait_port(self.port)
        return self.port
    
    async def stop(self):
        if self.proc:
            self.proc.terminate()
            await self.proc.wait()
            self.proc = None

# ---------- 工具 ----------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]

async def _wait_port(port: int, timeout=10):
    for _ in range(timeout * 10):
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 1)
            w.close()
            return
        except:
            await asyncio.sleep(0.1)
    raise RuntimeError("端口未就绪")

# ---------- 全局管理器 ----------
class NodeManager:
    def __init__(self):
        self.exts: Dict[str, NodeExtension] = {}

    async def start(self, ext_id: str) -> int:
        if ext_id not in self.exts:
            self.exts[ext_id] = NodeExtension(ext_id)
        return await self.exts[ext_id].start()

    async def stop(self, ext_id: str):
        if ext_id in self.exts:
            await self.exts[ext_id].stop()
            del self.exts[ext_id]

node_mgr = NodeManager()