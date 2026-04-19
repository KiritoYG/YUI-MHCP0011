import base64
import os
import re

import requests
from py.get_setting import load_settings,get_host,get_port,UPLOAD_FILES_DIR
from openai import AsyncClient
import uuid

from py.llm_tool import get_image_base64, get_image_media_type
async def pollinations_image(prompt: str, width=512, height=512, model="flux"):
    settings = await load_settings()
    
    # Check if the provided values are default ones, if so, override them with settings
    if width == 512:
        width = settings["text2imgSettings"]["pollinations_width"]
    if height == 512:
        height = settings["text2imgSettings"]["pollinations_height"]
    if model == "flux":
        model = settings["text2imgSettings"]["pollinations_model"]
    
    # Convert prompt into a URL-compatible format
    prompt = prompt.replace(" ", "%20")
    url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&nologo=true&enhance=true&private=true&safe=true"
    res_data = requests.get(url).content
    image_id = str(uuid.uuid4())
    # 将图片保存到本地UPLOAD_FILES_DIR，文件名为image_id，返回本地文件路径
    with open(f"{UPLOAD_FILES_DIR}/{image_id}.png", "wb") as f:
        f.write(res_data)
    return f"![image]({url})"

pollinations_image_tool = {
    "type": "function",
    "function": {
        "name": "pollinations_image",
        "description": "通过英文prompt生成图片，并返回markdown格式的图片链接，你必须直接以原markdown格式发给用户，用户才能直接看到图片。\n当你需要发送图片时，请将图片的URL放在markdown的图片标签中，例如：\n\n![图片名](图片URL)\n\n，图片markdown必须另起并且独占一行！",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "需要生成图片的英文prompt，例如：A little girl in a red hat。你可以尽可能的丰富你的prompt，以获得更好的效果",
                },
                "width": {
                    "type": "number",
                    "description": "图片宽度",
                    "default":512
                },
                "height": {
                    "type": "number",
                    "description": "图片高度",
                    "default": 512
                },
                "model": {
                    "type": "string",
                    "description": "使用的模型",
                    "default": "flux",
                    "enum": ["flux", "turbo"],
                }
            },
            "required": ["prompt"],
        },
    },
}

async def openai_image(prompt: str, size="auto"):
    settings = await load_settings()

    # Check if the provided values are default ones, if so, override them with settings
    if size == "auto":
        size = settings["text2imgSettings"]["size"]

    model = settings["text2imgSettings"]["model"]

    base_url = settings["text2imgSettings"]["base_url"]
    api_key = settings["text2imgSettings"]["api_key"]
    try:
        client = AsyncClient(api_key=api_key,base_url=base_url)
    
        response = await client.images.generate(prompt=prompt, size=size, model=model)
    except Exception as e:
        print(e)
        return f"ERROR: {e}"
    
    res_url = response.data[0].url
    res = f"![image]({res_url})"
    print(res)
    if res_url is None:
        res = response.data[0].b64_json
        HOST = get_host()
        if HOST == '0.0.0.0':
            HOST = '127.0.0.1'
        PORT = get_port()
        image_id = str(uuid.uuid4())
        # 将图片保存到本地UPLOAD_FILES_DIR，文件名为image_id，返回本地文件路径
        with open(f"{UPLOAD_FILES_DIR}/{image_id}.png", "wb") as f:
            f.write(base64.b64decode(res))
        res = f"![image](http://{HOST}:{PORT}/uploaded_files/{image_id}.png)"
    else:
        res_data = requests.get(res_url).content
        image_id = str(uuid.uuid4())
        # 将图片保存到本地UPLOAD_FILES_DIR，文件名为image_id，返回本地文件路径
        with open(f"{UPLOAD_FILES_DIR}/{image_id}.png", "wb") as f:
            f.write(res_data)
    return res
        
openai_image_tool = {
    "type": "function",
    "function": {
        "name": "openai_image",
        "description": "通过英文prompt生成图片，并返回markdown格式的图片链接，你必须直接以原markdown格式发给用户，用户才能直接看到图片。\n当你需要发送图片时，请将图片的URL放在markdown的图片标签中，例如：\n\n![图片名](图片URL)\n\n，图片markdown必须另起并且独占一行！",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "需要生成图片的英文prompt，例如：A little girl in a red hat。你可以尽可能的丰富你的prompt，以获得更好的效果",
                },
                "size": {
                    "type": "string",
                    "description": "图片大小，默认为auto",
                    "default": "auto", 
                    "enum": ["auto","1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"],
                }
            },
            "required": ["prompt"],
        },
    },
}

def process_image_content(text):
    # 正则表达式匹配 ![]() 中的内容
    pattern = r'!\[.*?\]\((.*?)\)'
    
    def replace_match(match):
        content = match.group(1)
        
        # 检查是否是 base64 数据
        if content.startswith('data:image'):
            # 提取 base64 部分（假设格式为 data:image/xxx;base64,实际数据）
            base64_data = content.split(',', 1)[1]
            HOST = get_host()
            if HOST == '0.0.0.0':
                HOST = '127.0.0.1'
            PORT = get_port()
            image_id = str(uuid.uuid4())
            
            # 确保上传目录存在
            os.makedirs(UPLOAD_FILES_DIR, exist_ok=True)
            
            # 保存图片到本地
            file_path = f"{UPLOAD_FILES_DIR}/{image_id}.png"
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(base64_data))
            
            # 返回新的图片链接
            return f"![image](http://{HOST}:{PORT}/uploaded_files/{image_id}.png)"
        else:
            # 如果是普通 URL，直接返回原内容
            return match.group(0)
    
    # 使用 re.sub 进行替换
    result = re.sub(pattern, replace_match, text)
    return result


async def openai_chat_image(prompt: str,img_url_list: list = []):
    settings = await load_settings()

    model = settings["text2imgSettings"]["model"]
    content = ""
    base_url = settings["text2imgSettings"]["base_url"]
    api_key = settings["text2imgSettings"]["api_key"]
    try:
        client = AsyncClient(api_key=api_key,base_url=base_url)
        if img_url_list:
            content = []
            for img_url in img_url_list:
                if img_url.startswith("http"):
                    base64_image = await get_image_base64(img_url)
                    media_type = await get_image_media_type(img_url)
                    img_url = f"data:{media_type};base64,{base64_image}"
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role":"user",
                    "content":content
                }
            ]
        )
    except Exception as e:
        print(e)
        return f"ERROR: {e}"
    
    if response:
        res = response.choices[0].message.content
        res = process_image_content(res)
    return res
        
openai_chat_image_tool = {
    "type": "function",
    "function": {
        "name": "openai_chat_image",
        "description": "通过英文prompt生成图片或者图片编辑，并返回markdown格式的图片链接，你必须直接以原markdown格式发给用户，用户才能直接看到图片。\n当你需要发送图片时，请将图片的URL放在markdown的图片标签中，例如：\n\n![图片名](图片URL)\n\n，图片markdown必须另起并且独占一行！",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "需要生成图片的英文prompt或者修改图片的prompt，例如：`A little girl in a red hat` 或者 `Change the girl's hat to white`。你可以尽可能的丰富你的prompt，以获得更好的效果",
                },
                "img_url_list": {
                    "type": "array",
                    "description": "执行图片编辑任务时的可选的字段，列表字段中每个元素都必须是图片URL，图片URL可以是用户上传的本地图片URL，格式类似于：http://127.0.0.1:3456/1.jpg ，也可以是一个公网上的图片URL",
                },
            },
            "required": ["prompt"],
        },
    },
}