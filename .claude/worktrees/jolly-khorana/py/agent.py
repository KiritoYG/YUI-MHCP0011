import json
import os
from pathlib import Path

def _get_project_config_path(cwd: str) -> Path:
    """获取 .party/config.json 的路径"""
    return Path(cwd) / ".party" / "config.json"

def is_tool_allowed_by_project_config(cwd: str, tool_name: str) -> bool:
    """
    检查项目级配置文件中是否已允许该工具
    """
    if not cwd:
        return False
    
    config_path = _get_project_config_path(cwd)
    if not config_path.exists():
        return False
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            allowed_tools = data.get("allowed_tools", [])
            return tool_name in allowed_tools
    except Exception as e:
        print(f"[Config Error] Failed to read .party config: {e}")
        return False

def add_tool_to_project_config(cwd: str, tool_name: str):
    """
    将工具添加到项目级允许列表中 (创建 .party/config.json)
    """
    if not cwd:
        return
        
    config_path = _get_project_config_path(cwd)
    party_dir = config_path.parent
    
    # 1. 确保文件夹存在
    if not party_dir.exists():
        party_dir.mkdir(parents=True, exist_ok=True)
        # 在 Windows 上隐藏文件夹 (可选)
        try:
            import ctypes
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(str(party_dir), FILE_ATTRIBUTE_HIDDEN)
        except:
            pass

    # 2. 读取现有配置
    data = {"allowed_tools": []}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            pass
            
    # 3. 更新并写入
    if tool_name not in data.get("allowed_tools", []):
        if "allowed_tools" not in data:
            data["allowed_tools"] = []
        data["allowed_tools"].append(tool_name)
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)