import shutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/uv")

# 1. 探针
@router.get("/probe")
def probe():
    return {"installed": shutil.which("uv") is not None}