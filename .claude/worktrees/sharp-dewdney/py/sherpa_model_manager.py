import asyncio
import os
import json
import uuid
from pathlib import Path
import httpx
import aiofiles
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# 假设您的应用中定义了一个默认的模型存储目录
# 注意：这里使用的是 DEFAULT_ASR_DIR
from py.get_setting import DEFAULT_ASR_DIR 

router = APIRouter(prefix="/sherpa-model")

# --- 模型配置 ---
MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue"
# Sherpa 运行时所需的关键文件列表
REQUIRED_FILES = ["model.int8.onnx", "tokens.txt"] 

MODELS = {
    "modelscope": {
        "url": "https://modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/resolve/master/model.int8.onnx",
        "tokens_url": "https://modelscope.cn/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/resolve/master/tokens.txt",
        # 必须定义 files_to_download 列表供下载接口使用
        "files_to_download": [
            {"filename": "model.int8.onnx", "url_key": "url", "progress_key": "model"},
            {"filename": "tokens.txt", "url_key": "tokens_url", "progress_key": "tokens"},
        ]
    },
    "huggingface": {
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/model.int8.onnx?download=true",
        "tokens_url": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/tokens.txt?download=true",
        "files_to_download": [
            {"filename": "model.int8.onnx", "url_key": "url", "progress_key": "model"},
            {"filename": "tokens.txt", "url_key": "tokens_url", "progress_key": "tokens"},
        ]
    }
}

# ---------- 工具函数 ----------
def get_model_dir() -> Path:
    """获取 Sherpa 模型在本地的完整路径"""
    return Path(DEFAULT_ASR_DIR) / MODEL_NAME

def model_exists() -> bool:
    """检查所有必需的模型文件是否存在"""
    d = get_model_dir()
    # 检查所有 REQUIRED_FILES 是否都存在于目录下
    return all((d / f).is_file() for f in REQUIRED_FILES)

async def download_file(url: str, dest: Path, progress_id: str):
    """
    异步下载单个文件并记录进度 (使用 DEFAULT_ASR_DIR)。
    所有文件写入操作都使用 aiofiles 保持异步。
    """
    tmp = dest.with_suffix(".downloading")
    progress_file_path = Path(DEFAULT_ASR_DIR) / f"{progress_id}.json"
    
    # 确保进度文件开始时存在 (异步写入)
    async with aiofiles.open(progress_file_path, mode='w') as p_file:
        await p_file.write(json.dumps({"done": 0, "total": 0}))

    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status() # 检查 HTTP 状态码
                total = int(resp.headers.get("content-length", 0))
                done = 0
                async with aiofiles.open(tmp, "wb") as f:
                    async for chunk in resp.aiter_bytes(1024 * 64):
                        await f.write(chunk)
                        done += len(chunk)
                        # 实时更新进度文件 (异步写入)
                        async with aiofiles.open(progress_file_path, mode='w') as p_file:
                            await p_file.write(
                                json.dumps({"done": done, "total": total, "filename": dest.name})
                            )
        
        # 将临时文件重命名为目标文件 (使用 asyncio.to_thread 处理同步的 Path.rename)
        await asyncio.to_thread(tmp.rename, dest)

        # 下载完成后，写入 complete 状态 (异步写入)
        async with aiofiles.open(progress_file_path, mode='w') as p_file:
            await p_file.write(
                json.dumps({"done": done, "total": done, "filename": dest.name, "complete": True})
            )
    except Exception as e:
        # 如果下载失败，记录错误信息 (异步写入)
        async with aiofiles.open(progress_file_path, mode='w') as p_file:
            await p_file.write(
                json.dumps({"error": str(e), "filename": dest.name, "failed": True})
            )
    finally:
        # 在下载任务结束后，不管成功与否，保留进度文件直到移除
        pass 

# ---------- 接口定义 ----------

@router.get("/status")
def status():
    """检查 Sherpa 模型文件是否存在"""
    return {"exists": model_exists(), "model": MODEL_NAME, "required_files": REQUIRED_FILES}

@router.delete("/remove")
def remove():
    """移除本地的 Sherpa 模型目录"""
    import shutil
    d = get_model_dir()
    if d.exists():
        shutil.rmtree(d)
    # 清理所有相关的进度文件 (使用 MODEL_NAME 前缀)
    for f in Path(DEFAULT_ASR_DIR).glob(f"{MODEL_NAME}_*.json"):
        f.unlink(missing_ok=True)
    return {"ok": True}

@router.get("/download/{source}")
async def download(source: str):
    """从指定源异步下载 Sherpa 模型和分词器文件，并流式传输进度"""
    if source not in MODELS:
        allowed_sources = list(MODELS.keys())
        raise HTTPException(status_code=400, detail=f"Bad source: only {', '.join(allowed_sources)} is supported.")
    if model_exists():
        raise HTTPException(status_code=400, detail="Model already exists.")

    model_subdir = get_model_dir()
    model_subdir.mkdir(parents=True, exist_ok=True)
    
    # 使用一个总的 ID 来追踪所有的下载任务
    master_progress_id = f"{MODEL_NAME}_{uuid.uuid4().hex}"
    
    # 创建所有下载任务
    tasks = []
    file_map = {} # 用于在生成器中查找每个文件进度的映射
    
    for item in MODELS[source]["files_to_download"]:
        filename = item["filename"]
        # 使用 item["url_key"] 从 MODELS[source] 中获取对应的 URL
        url = MODELS[source][item["url_key"]]
        progress_key = item["progress_key"]
        
        # 每个下载任务有一个唯一的 ID
        task_id = f"{master_progress_id}_{progress_key}" 
        dest_path = model_subdir / filename
        
        tasks.append(
            asyncio.create_task(
                download_file(url, dest_path, task_id)
            )
        )
        # 初始化 file_map 用于追踪状态
        file_map[progress_key] = {"id": task_id, "filename": filename, "done": 0, "total": 0, "complete": False, "failed": False}


    async def event_generator():
        # 监听所有文件的进度
        num_files = len(file_map)
        completed_files = 0
        
        # 清理下载进度文件（在任务完成后）
        def cleanup_progress_files():
            for key in file_map:
                try:
                    file_id = file_map[key].get('id')
                    if file_id:
                        progress_file = Path(DEFAULT_ASR_DIR) / f"{file_id}.json"
                        progress_file.unlink(missing_ok=True)
                except Exception as e:
                    # 安全地获取错误信息，避免触发异常对象的 __str__() 方法中的错误
                    try:
                        error_msg = str(e)
                    except:
                        error_msg = f"Error type: {type(e).__name__}"
                    
                    filename = file_map[key].get('filename', 'unknown')
                    print(f"Cleanup error for {filename}: {error_msg}")

        try:
            while completed_files < num_files:
                await asyncio.sleep(0.5)
                current_progress = {"status": "downloading", "files": []}
                completed_files = 0
                is_failed = False
                
                # 遍历所有文件，读取各自的进度文件
                for key in file_map:
                    file_info = file_map[key]
                    progress_file_path = Path(DEFAULT_ASR_DIR) / f"{file_info['id']}.json"
                    
                    try:
                        # 尝试异步读取进度文件内容 (修复：使用 asyncio.to_thread 避免阻塞事件循环)
                        file_content = await asyncio.to_thread(progress_file_path.read_text)
                        data = json.loads(file_content)
                        
                        file_info.update({
                            "done": data.get("done", 0),
                            "total": data.get("total", 0),
                            "complete": data.get("complete", False),
                            "failed": data.get("failed", False),
                            "error": data.get("error", None)
                        })
                        
                        if file_info["complete"]:
                            completed_files += 1
                        if file_info["failed"]:
                            is_failed = True
                        
                    except FileNotFoundError:
                        # 任务可能刚开始，进度文件尚未创建
                        pass
                    except json.JSONDecodeError:
                        # 进度文件可能正在写入中，忽略当前读取
                        pass
                    except Exception as e:
                        # 捕获其他可能的线程池或文件系统错误
                        print(f"Unexpected file read error for {file_info['filename']}: {e}")
                        
                    current_progress["files"].append({
                        "filename": file_info["filename"],
                        "done": file_info["done"],
                        "total": file_info["total"],
                        "complete": file_info["complete"],
                        "failed": file_info["failed"],
                        "error": file_info["error"]
                    })
                
                # 传输当前进度
                yield f"data: {json.dumps(current_progress)}\n\n"

                if is_failed:
                    current_progress["status"] = "failed"
                    yield f"data: {json.dumps(current_progress)}\n\n"
                    break # 退出循环

                if completed_files == num_files:
                    current_progress["status"] = "complete"
                    yield f"data: {json.dumps(current_progress)}\n\n"
                    break # 退出循环
                    
            # 最终清理
            cleanup_progress_files()
            yield "data: close\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            cleanup_progress_files()


    return StreamingResponse(event_generator(), media_type="text/event-stream")