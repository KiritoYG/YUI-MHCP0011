import requests
from typing import List, Optional, Union
from py.get_setting import load_settings


# 默认的基础 URL
DEFAULT_BASE_URL = "https://topics-after-party.zeabur.app"

async def get_random_topics(
    locale: str = "en-US",
    limit: int = 1,
    mood: Optional[str] = None,
    depth: Optional[int] = None,
    category: Optional[str] = None,
    exclude: Optional[Union[str, List[str]]] = None
) -> str:  # 注意：返回值类型提示从 dict 改为了 str
    """
    获取随机话题并返回格式化的 Markdown 文本
    """
    try:
        settings = await load_settings() # 假设这是你的配置加载逻辑
        base_url = settings["tools"]["randomTopic"].get("baseURL", DEFAULT_BASE_URL)
        endpoint = f"{base_url}/api/topic"
        
        if isinstance(exclude, list):
            exclude = ",".join(exclude)

        params = {
            "locale": locale,
            "limit": limit,
            "mood": mood,
            "depth": depth,
            "category": category,
            "exclude": exclude
        }
        
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        # 发送请求
        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        
        # --- 解析逻辑开始 ---
        res_json = response.json()
        
        # 1. 检查 API 状态码
        if res_json.get("code") != 200:
            return f"❌ 获取话题失败: API 返回错误代码 {res_json.get('code')}"

        data_list = res_json.get("data", [])
        
        # 2. 如果没有数据
        if not data_list:
            return "📭 未找到符合条件的话题。"

        # 3. 格式化为 Markdown
        md_output = []
        for idx, item in enumerate(data_list, 1):
            # 提取字段
            text = item.get("text", "")
            cat = item.get("category", "未知")
            tags = item.get("tags", [])
            follow_ups = item.get("follow_ups", [])
            # mood = item.get("mood", "") # 可选：是否需要展示情绪

            # 构建单个话题块
            # 格式：1. [分类] 话题内容
            topic_str = f"\n\n{idx}. **[{cat}]** {text}"
            
            # 添加标签 (可选)
            if tags:
                tag_str = " ".join([f"`#{t}`" for t in tags])
                topic_str += f"\n\n   > 🏷️ {tag_str}"
            
            # 添加追问 (可选)
            if follow_ups:
                topic_str += "\n\n   > 🗣️ **追问参考**："
                for fu in follow_ups:
                    topic_str += f"\n\n   > - {fu}"

            md_output.append(topic_str)

        # 用双换行连接，保持段落间距
        return "\n\n".join(md_output)
        # --- 解析逻辑结束 ---

    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return f"⚠️ 网络请求错误: {str(e)}"
    except Exception as e:
        return f"⚠️ 处理数据时发生错误: {str(e)}"
    
async def get_categories(
    locale: str = "en-US"
) -> List[str]:
    """
    获取分类列表 (Get Category List)
    
    Args:
        locale (str): 返回分类名称的语言，可选 'zh-CN' 或 'en-US'。默认 'en-US'。
        base_url (str): API 基础地址。

    Returns:
        List[str]: 分类名称列表。
    """
    try:
        settings = await load_settings()

        base_url = settings["tools"]["randomTopic"].get("baseURL", DEFAULT_BASE_URL)
        endpoint = f"{base_url}/api/categories"
        
        params = {
            "locale": locale
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
        return []
    

random_topics_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_random_topics",
            "description": "获取随机的聊天话题、破冰问题或深度对话主题。当用户想要开始一段对话、感到无聊、或者想要深入了解对方时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "locale": {
                        "type": "string",
                        "enum": ["zh-CN", "en-US"],
                        "description": "话题的语言环境。中文请使用 'zh-CN'，英文使用 'en-US'。默认为 'en-US'。",
                        "default": "en-US"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "一次获取的话题数量，默认为 1。",
                        "default": 1
                    },
                    "mood": {
                        "type": "string",
                        "enum": ["positive", "neutral", "curious", "flirty"],
                        "description": "话题的情绪基调。positive: 积极向上; neutral: 中性/一般; curious: 好奇/探索; flirty: 暧昧/调情。"
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "话题的深度等级 (1-5)。1 为轻松闲聊，5 为深度灵魂拷问。"
                    },
                    "category": {
                        "type": "string",
                        "description": "特定的话题分类（例如 'Life', 'Love' 等）。建议先调用 get_categories 获取可用分类。"
                    }
                },
                "required": [] 
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_categories",
            "description": "获取当前可用的话题分类列表。在用户想要选择特定类型的聊天话题时，先调用此函数查看有哪些分类。",
            "parameters": {
                "type": "object",
                "properties": {
                    "locale": {
                        "type": "string",
                        "enum": ["zh-CN", "en-US"],
                        "description": "分类名称的语言。中文请使用 'zh-CN'，英文使用 'en-US'。",
                        "default": "en-US"
                    }
                },
                "required": []
            }
        }
    }
]    