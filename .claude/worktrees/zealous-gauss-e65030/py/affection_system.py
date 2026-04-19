import os
import json
import re
import asyncio
from py.get_setting import USER_DATA_DIR

# 存储好感度数据的目录和文件
AFFECTION_DIR = os.path.join(USER_DATA_DIR, 'affection')
AFFECTION_FILE = os.path.join(AFFECTION_DIR, 'affection_data.json')

async def load_affection_data():
    """读取用户好感度数据"""
    os.makedirs(AFFECTION_DIR, exist_ok=True)
    if not os.path.exists(AFFECTION_FILE):
        return {}
    try:
        # 使用 asyncio.to_thread 防止阻塞事件循环
        def _read():
            with open(AFFECTION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return await asyncio.to_thread(_read)
    except Exception as e:
        print(f"[Affection] 读取数据失败: {e}")
        return {}

async def save_affection_data(data):
    """保存用户好感度数据"""
    os.makedirs(AFFECTION_DIR, exist_ok=True)
    try:
        def _write():
            with open(AFFECTION_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        await asyncio.to_thread(_write)
    except Exception as e:
        print(f"[Affection] 保存数据失败: {e}")

async def extract_and_update_affection(full_content):
    """从AI完整的回复中提取 <user=xxx love=xxx> 并更新数据"""
    if not full_content:
        return
    
    # 正则匹配：查找 <user=用户名 属性1=数值 属性2=数值>
    # 兼容带空格的情况，如 <user=派酱 love=12 familiarity=15>
    match = re.search(r"<user=([^\s>]+)\s+(.+?)>", full_content)
    if not match:
        return

    user_name = match.group(1)
    stats_str = match.group(2)

    # 提取所有的 属性=数值
    # 支持中文属性名、负数等
    stat_matches = re.findall(r"([a-zA-Z0-9_\u4e00-\u9fa5]+)\s*=\s*(-?\d+)", stats_str)
    
    if stat_matches:
        new_stats = {k: int(v) for k, v in stat_matches}
        
        # 更新到 JSON
        data = await load_affection_data()
        if user_name not in data:
            data[user_name] = {}
        
        data[user_name].update(new_stats)
        await save_affection_data(data)
        print(f"✨ [好感度系统] 用户 {user_name} 状态已更新: {new_stats}")