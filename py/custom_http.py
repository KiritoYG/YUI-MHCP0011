import aiohttp
import json

# 安全解析 JSON 的函数（用于 headers 是字符串的情况）
def safe_json_loads(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}

async def fetch_custom_http(method, url, headers=None, body=None):
    # 处理 headers
    if headers is None or headers == "":
        headers = {}
    elif isinstance(headers, str):
        print(f'headers: {headers}')
        headers = safe_json_loads(headers)

    # 自动处理 Content-Type，默认为 application/json
    content_type = headers.get('Content-Type', 'application/json')

    # 准备参数
    kwargs = {
        'headers': headers,
    }

    # 根据 Content-Type 决定使用 data 还是 json
    if content_type == 'application/json':
        kwargs['json'] = body
    else:
        kwargs['data'] = body

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as response:
                print(f'Status: {response.status}')
                response_text = await response.text()
                print(f'Response: {response_text}')
                return response_text
    except Exception as e:
        print(f'Error: {e}')
        return f'Error: {e}'