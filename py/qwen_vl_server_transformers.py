"""
Qwen VL (2.5 / 2 / 3 系列) Transformers 后端 —— 迷你 OpenAI 兼容服务

用途：当 vLLM 不可用时（常见于 Windows），用 transformers 直接托管本地 Qwen VL
模型，暴露 OpenAI `/v1/chat/completions` 与 `/v1/models`，供 server.py 的
主聊天 / 桌面截图翻译 / 摄像头视觉 等流程调用。

启动示例（由 qwen_vl_launcher 自动拼接）：

    python py/qwen_vl_server_transformers.py \
        --model-path G:\Qwen2.5-VL-3B-Instruct \
        --host 127.0.0.1 --port 8001 \
        --served-name Qwen2.5-VL-3B \
        --dtype bfloat16 --max-new-tokens 512

接收的 messages 支持：
  - content 为纯字符串
  - content 为 list，每项 {"type":"text","text":...} 或
    {"type":"image_url","image_url":{"url":"http(s)://..."|"data:...;base64,..."|本地路径}}

已序列化：同一时刻只跑一个请求（单卡单模型），避免显存抖动。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel


# ---------- 模型加载 ----------

_model = None
_processor = None
_served_name = "Qwen-VL"
_dtype = torch.bfloat16
_device = "cuda" if torch.cuda.is_available() else "cpu"
_max_new_tokens_default = 512
_gen_lock = asyncio.Lock()


def load_model(model_path: str, dtype: str = "bfloat16"):
    """加载 Qwen2.5-VL / Qwen2-VL。transformers 5.x 统一走 AutoModel*."""
    global _model, _processor, _dtype
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "auto": "auto",
    }
    _dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"[qwen-vl-server] loading {model_path} (dtype={dtype}, device={_device})...", flush=True)
    t0 = time.time()
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=_dtype,
        device_map=_device,
    )
    _processor = AutoProcessor.from_pretrained(model_path)
    _model.eval()
    print(f"[qwen-vl-server] model loaded in {time.time()-t0:.1f}s", flush=True)


# ---------- 图像解析 ----------

_data_url_re = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


async def _resolve_image(url_or_path: str) -> Image.Image:
    """把 OpenAI image_url 里的 url 变成 PIL.Image：支持 data:、http(s)://、本地路径。"""
    if not url_or_path:
        raise ValueError("empty image url")

    # data:image/xxx;base64,....
    m = _data_url_re.match(url_or_path.strip())
    if m:
        raw = base64.b64decode(m.group("data"))
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # http(s)://
    if url_or_path.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url_or_path)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")

    # 本地路径
    if os.path.exists(url_or_path):
        return Image.open(url_or_path).convert("RGB")

    raise ValueError(f"cannot resolve image url: {url_or_path[:120]}")


async def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 OpenAI 消息转成 Qwen processor.apply_chat_template 能吃的格式。
    - text 不变
    - image_url -> {"type":"image","image":<PIL>}  （processor 支持 PIL 对象）
    """
    norm = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if content is None:
            norm.append({"role": role, "content": ""})
            continue
        if isinstance(content, str):
            norm.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            new_items = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text":
                    new_items.append({"type": "text", "text": item.get("text", "")})
                elif t == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    try:
                        pil = await _resolve_image(url)
                        new_items.append({"type": "image", "image": pil})
                    except Exception as e:
                        # 图片解析失败，降级为文字
                        new_items.append({"type": "text", "text": f"[image load failed: {e}]"})
                # 其它类型忽略
            norm.append({"role": role, "content": new_items})
            continue
        # 其它类型兜底
        norm.append({"role": role, "content": str(content)})
    return norm


# ---------- 生成 ----------

async def _generate(messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    norm = await _normalize_messages(messages)
    # 抽出 PIL 图像列表（processor 需要单独的 images 参数）
    images: List[Image.Image] = []
    for m in norm:
        c = m.get("content")
        if isinstance(c, list):
            for it in c:
                if isinstance(it, dict) and it.get("type") == "image":
                    images.append(it["image"])

    text = _processor.apply_chat_template(norm, tokenize=False, add_generation_prompt=True)

    async with _gen_lock:
        inputs = _processor(
            text=[text],
            images=images if images else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(_device)

        t0 = time.time()
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature is not None and temperature > 0),
                temperature=max(temperature or 0.0, 1e-5) if (temperature and temperature > 0) else 1.0,
                top_p=top_p if top_p is not None else 1.0,
            )
        gen_ids = out[:, inputs.input_ids.shape[1]:]
        completion = _processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        elapsed = time.time() - t0
        n_out = int(gen_ids.shape[1])
        print(f"[qwen-vl-server] generated {n_out} tokens in {elapsed:.2f}s ({n_out/max(elapsed,1e-3):.1f} tok/s)", flush=True)

        # 统计 token
        prompt_tokens = int(inputs.input_ids.shape[1])

    return {
        "text": completion.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": n_out,
    }


# ---------- FastAPI ----------

app = FastAPI(title="Qwen VL Transformers Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _served_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.get("/health")
async def health():
    return {"ok": True, "model": _served_name, "device": _device, "loaded": _model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    max_new = int(body.max_tokens or _max_new_tokens_default)
    try:
        result = await _generate(body.messages, max_new, body.temperature or 0.0, body.top_p or 1.0)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[qwen-vl-server] generation error:\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

    resp_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": result["text"]},
        "finish_reason": "stop",
    }
    if body.stream:
        # 简易"伪流"：一次 chunk + DONE
        from fastapi.responses import StreamingResponse
        import json

        async def gen():
            chunk = {
                "id": resp_id, "object": "chat.completion.chunk", "created": created,
                "model": _served_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": _served_name,
        "choices": [choice],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


# ---------- 入口 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--served-name", default="Qwen2.5-VL-3B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    global _served_name, _max_new_tokens_default
    _served_name = args.served_name
    _max_new_tokens_default = args.max_new_tokens

    load_model(args.model_path, args.dtype)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
