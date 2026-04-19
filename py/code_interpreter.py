from e2b_code_interpreter import Sandbox
import asyncio
from concurrent.futures import ThreadPoolExecutor
from py.get_setting import load_settings

async def e2b_code(code: str, language: str = "Python") -> str:
    settings = await load_settings()
    e2b_api_key = settings["codeSettings"]["e2b_api_key"]
    executor = ThreadPoolExecutor()
    def run_in_sandbox():
        with Sandbox(api_key=e2b_api_key) as sandbox:
            execution = sandbox.run_code(code,language=language)
            return execution.logs

    # 使用线程池执行同步代码，防止阻塞事件循环
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_in_sandbox)
    return str(result)

import asyncio
from aiohttp import ClientSession


async def local_run_code(code: str, language: str = "python") -> str:
    settings = await load_settings()
    url = settings["codeSettings"]["sandbox_url"].strip("/") + "/run_code"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "code": code,
        "language": language
    }

    async with ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            # 获取响应文本
            result = await response.text()
            return result

e2b_code_tool = {
    "type": "function",
    "function": {
        "name": "e2b_code",
        "description": "执行代码，工具只会返回stdout和stderr。请将你要查看的答案输出到stdout。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "需要执行的代码，例如：print('Hello, World!')，不要包含markdown的代码块标记！只有输入可运行的代码字符串。",
                },
                "language": {
                    "type": "string",
                    "description": "代码语言。",
                    "enum": ["python", "js", "ts", "r", "java", "bash"],
                    "default": "python"
                }
            },
            "required": ["code"],
        },
    },
}

local_run_code_tool = {
  "type": "function",
  "function": {
    "name": "local_run_code",
    "description": "执行代码，工具只会返回stdout和stderr。请将你要查看的答案输出到stdout。",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "需要执行的代码，例如：print('Hello, World!')，不要包含markdown的代码块标记！只有输入可运行的代码字符串。工具只会返回stdout和stderr。请将你要查看的答案放在print()中，不要放在其他地方。"
        },
        "language": {
          "type": "string",
          "description": "代码语言。",
          "enum": [
            "python", "cpp", "nodejs", "go", "go_test", "java", "php", "csharp",
            "bash", "typescript", "sql", "rust", "cuda", "lua", "R", "perl",
            "D_ut", "ruby", "scala", "julia", "pytest", "junit", "kotlin_script",
            "jest", "verilog", "python_gpu", "lean", "swift", "racket"
          ],
          "default": "python"
        }
      },
      "required": ["code"]
    }
  }
}