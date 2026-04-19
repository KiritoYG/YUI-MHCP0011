from fastapi import APIRouter, Body, HTTPException
from typing import Dict, Any
from py.affection_system import load_affection_data, save_affection_data

# 创建羁绊系统的数据路由
router = APIRouter(prefix="/api/affection", tags=["Affection System"])

@router.get("/get_data")
async def get_affection_data_api():
    """
    获取所有用户的羁绊数据
    返回格式: {"小包": {"love": 10, "Familiarity": 5}, "张三": {"love": 2}}
    """
    try:
        data = await load_affection_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取羁绊数据失败: {str(e)}")

@router.post("/save_data")
async def save_affection_data_api(data: Dict[str, Any] = Body(...)):
    """
    全量保存羁绊数据 (覆盖保存)
    接收格式: {"小包": {"love": 10, "Familiarity": 5}}
    """
    try:
        await save_affection_data(data)
        return {"status": "success", "message": "羁绊数据保存成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存羁绊数据失败: {str(e)}")