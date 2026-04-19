# mcp_client_fixed.py
import json
import asyncio
import logging
import shutil
from typing import Dict, Any, AsyncIterator, Optional

import anyio
from mcp import ClientSession
from mcp.client.stdio   import stdio_client
from mcp.client.sse     import sse_client
from mcp.client.websocket import websocket_client
from mcp.client.streamable_http import streamablehttp_client
from contextlib import AsyncExitStack, asynccontextmanager

# ---------- 工具 ----------
def get_command_path(command_name: str, default_command: str = "uv") -> str:
    path = shutil.which(command_name) or shutil.which(default_command)
    if not path:
        raise FileNotFoundError(f"未找到 {command_name} 或 {default_command}")
    return path

# ---------- 连接管理 ----------
class ConnectionManager:
    def __init__(self) -> None:
        self.session: Optional[ClientSession] = None
        self.tools: list[str] = []

    @asynccontextmanager
    async def connect(self, config: dict) -> AsyncIterator["ConnectionManager"]:
        async with AsyncExitStack() as stack:
            # 1. 建立传输层
            if "command" in config:
                from mcp.client.stdio import StdioServerParameters
                server_params = StdioServerParameters(
                    command=get_command_path(config["command"]),
                    args=config.get("args", []),
                    env=config.get("env"),
                )
                read, write = await stack.enter_async_context(stdio_client(server_params))
            else:
                mcptype = config.get("type", "ws")
                if "streamable" in mcptype:
                    mcptype = "streamablehttp"
                client_map = {
                    "ws": websocket_client,
                    "sse": sse_client,
                    "streamablehttp": streamablehttp_client,
                }
                headers = config.get("headers", {})
                client = client_map[mcptype](
                    config["url"], headers=headers
                ) if headers else client_map[mcptype](config["url"])

                transport = await stack.enter_async_context(client)

                # ---------- END ----------

                if mcptype == "streamablehttp":
                    read, write, _ = transport
                else:
                    read, write = transport
                # ---------- 首包校验（仅 SSE 需要） ----------
                if mcptype == "sse":
                    try:
                        # 非阻塞读 1 条消息，超时 3 秒
                        with anyio.move_on_after(3):
                            await read.receive()
                    except anyio.EndOfStream:
                        # 服务器立刻关闭，说明失败
                        raise RuntimeError("SSE stream closed immediately")
                    except Exception as e:
                        raise RuntimeError(f"SSE initial handshake failed: {e}") from e
            # 2. 建立会话
            self.session = await stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            self.tools = [t.name for t in (await self.session.list_tools()).tools]
            logging.info("Connected to MCP server. Tools: %s", self.tools)

            yield self


# ---------- 客户端 ----------
class McpClient:
    def __init__(self) -> None:
        self._conn: Optional[ConnectionManager] = None
        self._config: Optional[dict] = None
        self._lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._on_failure_callback: Optional[callable] = None  # 新增：失败回调
        self._tools: list[str] = []
        self._tools_list = []

    async def initialize(self, server_name: str, server_config: dict, on_failure_callback: Optional[callable] = None) -> None:
        """非阻塞初始化：拉起连接监控协程"""
        self._config = server_config
        self._on_failure_callback = on_failure_callback  # 设置回调
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._connection_monitor())

    async def close(self) -> None:
        self._shutdown = True
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _connection_monitor(self) -> None:
        """持续重连逻辑：仅在一个协程里管理 AsyncExitStack"""
        while not self._shutdown:
            try:
                async with ConnectionManager().connect(self._config) as conn:
                    async with self._lock:
                        self._conn = conn
                    # 心跳检测
                    while not self._shutdown:
                        try:
                            await asyncio.wait_for(self._conn.session.send_ping(), timeout=3)
                        except Exception:
                            break  # 断线，跳出 inner loop
                        await asyncio.sleep(30)
            except Exception as e:
                logging.exception("Connection failed, will retry: %s", e)
                if self._on_failure_callback:
                    await self._on_failure_callback(str(e))  # 调用回调
            finally:
                async with self._lock:
                    self._conn = None
            if not self._shutdown:
                await asyncio.sleep(5)

    # ---------- 外部 API ----------
    async def get_openai_functions(self,disable_tools=[]):
        async with self._lock:
            if not self._conn or not self._conn.session:
                return []
            tools = (await self._conn.session.list_tools()).tools
            self._tools = [t.name for t in tools]
            self._tools_list = [{"name": t.name, "description": t.description,"enabled":True} for t in tools]
            tools_list = []
            for t in tools:
                if t.name not in disable_tools:
                    tools_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.inputSchema,
                            },
                        }
                    )

            return tools_list

    async def call_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Any:
        async with self._lock:
            if not self._conn or not self._conn.session:
                return None
            try:
                return await self._conn.session.call_tool(tool_name, tool_params)
            except Exception as e:
                logging.error("Failed to call tool %s: %s", tool_name, e)
                return "Failed to call tool %s: %s" % (tool_name, e)


# ---------- 使用示例 ----------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    async def main():
        client = McpClient()
        await client.initialize(
            "example",
            {
                "type": "sse",
                "url": "http://127.0.0.1:8000/mcp",
            },
        )
        await asyncio.sleep(2)
        funcs = await client.get_openai_functions()
        print("OpenAI functions:", funcs)
        await asyncio.sleep(30)  # 保持连接
        await client.close()

    asyncio.run(main())