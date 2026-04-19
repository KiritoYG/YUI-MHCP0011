import base64
import logging
import os
import random
import re
import tempfile
import time

import requests
from py.get_setting import UPLOAD_FILES_DIR, load_settings


async def upload_image_host(url):
    settings = await load_settings()
    # 检查图床功能是否开启
    if not settings["BotConfig"]["imgHost_enabled"]:
        return url
    
    # 处理本地文件上传
    if 'uploaded_files' in url:
        file_name = url.split("/")[-1]
        file_path = os.path.join(UPLOAD_FILES_DIR, file_name)
        return await _upload_file(settings, file_path)
    
    # 处理外部URL上传
    try:
        # 下载图片到临时文件
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            logging.error(f"下载图片失败: HTTP {response.status_code} - {url}")
            return url
        
        # 获取内容类型和扩展名
        content_type = response.headers.get('Content-Type', '')
        ext_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/svg+xml': '.svg',
        }
        ext = ext_map.get(content_type.split(';')[0], '.bin')
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            suffix=ext, 
            dir=UPLOAD_FILES_DIR,
            delete=False
        ) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            file_path = tmp_file.name
        
        logging.info(f"下载外部图片到临时文件: {file_path}")
        return await _upload_file(settings, file_path)
    
    except Exception as e:
        logging.error(f"处理外部图片异常: {str(e)}")
        return url

async def _upload_file(settings, file_path):
    """实际执行文件上传的内部函数"""
    # 确保文件存在
    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        return f"文件不存在: {file_path}"
    
    file_name = os.path.basename(file_path)
    is_temp_file = 'tmp' in file_path  # 标记临时文件
    
    try:

        # EasyImage图床处理
        if settings["BotConfig"]["imgHost"] == "easyImage2":
            EI2_url = settings["BotConfig"]["EI2_base_url"]
            EI2_token = settings["BotConfig"]["EI2_api_key"]
            
            with open(file_path, "rb") as f:
                files = {"image": (file_name, f)}
                data = {"token": EI2_token}
                response = requests.post(EI2_url, data=data, files=files)
            
            if response.status_code == 200:
                return response.json().get("url")
            else:
                logging.error(f"EasyImage上传失败: {response.status_code}")
                return f"EasyImage上传失败: {response.status_code}"

        # 未知图床类型
        else:
            logging.warning(f"未知图床类型: {settings['BotConfig']['imgHost']}")
            return f"未知图床类型: {settings['BotConfig']['imgHost']}"
    
    except Exception as e:
        logging.error(f"图床上传异常: {str(e)}")
        return f"图床上传异常: {str(e)}"
    
    finally:
        # 清理临时文件
        if is_temp_file and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"已清理临时文件: {file_path}")
            except Exception as e:
                logging.error(f"清理临时文件失败: {str(e)}")