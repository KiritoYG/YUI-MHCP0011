import shutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/node")

@router.get("/probe")
def probe():
    return {"installed": shutil.which("node") is not None}