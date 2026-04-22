"""
Qwen3.5-VL 本地 OpenAI 兼容推理服务（纯 transformers，Windows 原生可跑）

- 不依赖 vLLM
- 提供 POST /v1/chat/completions（多模态文本+图像，OpenAI 格式）
- 提供 GET /v1/models、GET /health
- 作为独立脚本由 qwen_vl_launcher 以子进程启动

命令行兼容 vLLM 的部分常用参数（未识别参数会被静默忽略，便于 GUI 复用）：
  --model <path>
  --host 127.0.0.1
  --port 8000
  --served-model-name Qwen3.5-VL
  --trust-remote-code
  --dtype auto|bfloat16|float16|float32
  --max-new-tokens 1024
  --max-model-len ...      (accepted, ignored)
  --gpu-memory-utilization (accepted, ignored)
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
import uuid
import threading
from typing import List, Optional, Union, Any

try:
    import torch
except Exception as e:
    print(f"[qwen_vl_server] torch import failed: {e}", flush=True)
    raise

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except Exception as e:
    print(f"[qwen_vl_server] fastapi/uvicorn/pydantic missing: {e}", flush=True)
    raise

try:
    from PIL import Image
except Exception as e:
    print(f"[qwen_vl_server] Pillow missing: {e}", flush=True)
    raise

try:
    import httpx
except Exception:
    httpx = None


app = FastAPI(title="Qwen3.5-VL Local")

_state = {
    "model": None,
    "processor": None,
    "served_name": "Qwen3.5-VL",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "load_error": None,
    "loading": True,
    "default_max_new_tokens": 1024,
}


# ----------------------------- 辅助 -----------------------------
def _load_image(src: str) -> Image.Image:
    if src.startswith("data:"):
        _, b64 = src.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if src.startswith("http://") or src.startswith("https://"):
        if httpx is None:
            raise RuntimeError("httpx not installed; cannot fetch remote image")
        r = httpx.get(src, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    # 本地路径
    return Image.open(src).convert("RGB")


def _convert_messages(messages: List[Any]):
    """把 OpenAI 风格 messages 转成 Qwen-VL chat_template 格式，并抽出图片列表。"""
    converted = []
    images: List[Image.Image] = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue

        parts = []
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if t == "text":
                parts.append({"type": "text", "text": c.get("text", "")})
            elif t == "image_url":
                iu = c.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    img = _load_image(url)
                    images.append(img)
                    parts.append({"type": "image"})
            elif t == "image":
                src = c.get("image") or c.get("url")
                if src:
                    img = _load_image(src) if isinstance(src, str) else src
                    images.append(img)
                    parts.append({"type": "image"})
        converted.append({"role": role, "content": parts})
    return converted, images


# ----------------------------- Schemas -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


# ----------------------------- Endpoints -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "loading": _state["loading"],
        "load_error": _state["load_error"],
        "model_loaded": _state["model"] is not None,
        "device": _state["device"],
    }


@app.get("/v1/models")
def list_models():
    if _state["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not ready", "loading": _state["loading"], "load_error": _state["load_error"]},
        )
    return {
        "object": "list",
        "data": [
            {
                "id": _state["served_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if _state["model"] is None or _state["processor"] is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not ready", "loading": _state["loading"], "load_error": _state["load_error"]},
        )

    try:
        messages_q, images = _convert_messages([m.dict() for m in req.messages])
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"message/image parse failed: {e}"})

    processor = _state["processor"]
    model = _state["model"]

    try:
        text = processor.apply_chat_template(
            messages_q, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"chat_template failed: {e}"})

    try:
        call_kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
        if images:
            call_kwargs["images"] = images
        inputs = processor(**call_kwargs)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"processor call failed: {e}"})

    max_new = int(req.max_tokens) if req.max_tokens else _state["default_max_new_tokens"]
    do_sample = (req.temperature or 0) > 0.0
    gen_kwargs = {"max_new_tokens": max_new, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = float(req.temperature)
        gen_kwargs["top_p"] = float(req.top_p or 1.0)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"generate failed: {e}"})

    input_len = inputs["input_ids"].shape[1]
    trimmed = output_ids[:, input_len:]
    response_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _state["served_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(input_len),
            "completion_tokens": int(trimmed.shape[1]),
            "total_tokens": int(output_ids.shape[1]),
        },
    }


# ----------------------------- 模型加载 -----------------------------
def _resolve_dtype(dtype_str: str):
    s = (dtype_str or "auto").lower()
    if s in ("auto",):
        return torch.bfloat16 if _state["device"] == "cuda" else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(s, torch.bfloat16)


def _load_model_thread(model_path: str, served_name: str, dtype_str: str) -> None:
    try:
        _state["served_name"] = served_name
        print(f"[qwen_vl_server] device={_state['device']} dtype={dtype_str}", flush=True)
        print(f"[qwen_vl_server] loading processor from {model_path}", flush=True)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"[qwen_vl_server] loading model from {model_path} ...", flush=True)
        torch_dtype = _resolve_dtype(dtype_str)

        # 先尝试 AutoModelForImageTextToText（transformers 新增的 VL 通用类）
        model = None
        last_err: Optional[Exception] = None
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
            )
        except Exception as e1:
            last_err = e1
            print(f"[qwen_vl_server] AutoModelForImageTextToText failed: {e1}; fallback to AutoModel", flush=True)
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
                )
            except Exception as e2:
                print(f"[qwen_vl_server] AutoModel failed too: {e2}", flush=True)
                raise RuntimeError(
                    f"model load failed. primary={last_err}; fallback={e2}. "
                    f"Hint: transformers version may be too old for Qwen3.5 architecture. "
                    f"Try: pip install -U 'transformers>=4.57.0.dev0' or install from source."
                )

        model.eval()
        _state["processor"] = processor
        _state["model"] = model
        _state["loading"] = False
        print(f"[qwen_vl_server] READY served-name={served_name}", flush=True)
    except Exception as e:
        _state["load_error"] = str(e)
        _state["loading"] = False
        print(f"[qwen_vl_server] LOAD FAILED: {e}", flush=True)


# ----------------------------- 入口 -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default="Qwen3.5-VL")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--trust-remote-code", action="store_true")
    # vLLM 兼容但在此处忽略的参数：
    p.add_argument("--max-model-len", default=None)
    p.add_argument("--gpu-memory-utilization", default=None)
    p.add_argument("--tensor-parallel-size", default=None)
    p.add_argument("--quantization", default=None)

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[qwen_vl_server] ignored args: {unknown}", flush=True)

    _state["default_max_new_tokens"] = int(args.max_new_tokens)

    # 先把 uvicorn 拉起来（/health 立刻可用），模型在后台线程加载
    t = threading.Thread(
        target=_load_model_thread,
        args=(args.model, args.served_model_name, args.dtype),
        daemon=True,
    )
    t.start()

    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
