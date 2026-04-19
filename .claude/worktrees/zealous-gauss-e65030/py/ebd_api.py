from typing import Optional
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from py.load_files import sanitize_url

router = APIRouter(prefix="/api", tags=["extra"])

class EmbeddingDimsRequest(BaseModel):
    api_key: str
    base_url: Optional[HttpUrl] = None   # 留空时走官方 https://api.openai.com/v1
    model: str

class EmbeddingDimsResponse(BaseModel):
    dims: int

@router.post("/embedding_dims", response_model=EmbeddingDimsResponse)
async def get_embedding_dims(req: EmbeddingDimsRequest):
    """
    用任意句子调一次嵌入接口，把返回向量的长度作为维度返回。
    兼容 OpenAI 官方以及任何「接口格式一致」的代理。
    """
    url = sanitize_url(
        input_url=req.base_url, 
        default_base="https://api.openai.com/v1", 
        endpoint="/embeddings"
    )

    payload = {"model": req.model, "input": "test"}
    headers = {"Authorization": f"Bearer {req.api_key}"}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
        except Exception as e:
            raise HTTPException(502, detail=f"请求嵌入接口失败: {e}")

    if r.status_code != 200:
        raise HTTPException(
            status_code=r.status_code,
            detail=f"上游返回错误: {r.text}"
        )

    try:
        vec = r.json()["data"][0]["embedding"]
    except (KeyError, IndexError):
        raise HTTPException(502, detail="上游返回格式异常，无法解析 embedding")

    return EmbeddingDimsResponse(dims=len(vec))