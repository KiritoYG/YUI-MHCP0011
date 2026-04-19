import asyncio
import socket
import ssl
import time
from typing import Callable, Optional

class SimpleTwitchChat:
    """
    仅负责“收包-解析-回调”，不再管事件循环。
    生命周期由外部 start_twitch_task / stop_twitch_task 控制。
    """
    def __init__(self, access_token: str, channel: str):
        self.access_token = access_token.replace("oauth:", "")
        self.channel = channel.lower().lstrip("#")
        self._sock: Optional[socket.socket] = None
        self._callback: Optional[Callable[[str, str, str], None]] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ---------- 外部调用 ----------
    def set_callback(self, cb: Callable[[str, str, str], None]):
        self._callback = cb

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._close_socket()

    # ---------- 内部 ----------
    async def _listen_loop(self):
        reconnect_delay = 5
        while self._running:
            try:
                await self._connect_and_read()
                reconnect_delay = 5
            except Exception as exc:
                if not self._running:
                    break
                print(f"[Twitch] 连接异常: {exc}，{reconnect_delay}s 后重连")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    async def _connect_and_read(self):
        ctx = ssl.create_default_context()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        self._sock = ctx.wrap_socket(sock, server_hostname="irc.chat.twitch.tv")
        self._sock.connect(("irc.chat.twitch.tv", 6697))
        self._sock.settimeout(None)

        # 认证
        self._send(f"CAP REQ :twitch.tv/tags twitch.tv/commands")
        self._send(f"PASS oauth:{self.access_token}")
        self._send(f"NICK justinfan12345")
        self._send(f"JOIN #{self.channel}")

        buffer = ""
        while self._running:
            data = await asyncio.get_event_loop().sock_recv(self._sock, 4096)
            if not data:
                raise ConnectionAbortedError("服务器关闭连接")
            buffer += data.decode("utf-8", errors="ignore")
            while "\r\n" in buffer:
                line, buffer = buffer.split("\r\n", 1)
                if line:
                    self._handle_line(line)

    def _handle_line(self, line: str):
        if line.startswith("PING"):
            self._send("PONG " + line[4:])
            return
        if "PRIVMSG" not in line:
            return

        # 1. 提取标签段（@ 开头，空格结束）
        tags = {}
        if line.startswith("@"):
            tag_str, _, line = line[1:].partition(" ")
            for kv in tag_str.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    tags[k] = v

        # 2. 用户名：优先 display-name，其次 user-id，最后 login
        user = (
            tags.get("display-name") or
            tags.get("user-id") or
            line.split("!", 1)[0]  # 最末兜底
        ).strip()

        # 3. 频道名
        try:
            _, _, rest = line.partition("PRIVMSG #")
            channel = rest.split(" ", 1)[0].lower().lstrip("#")
        except Exception:
            return

        # 4. 消息内容
        try:
            msg = line.split(" :", maxsplit=1)[1]
        except Exception:
            return

        # 5. 回调
        if self._callback:
            asyncio.create_task(
                self._callback(channel, user, msg)
                if asyncio.iscoroutinefunction(self._callback)
                else asyncio.get_event_loop().run_in_executor(
                    None, self._callback, channel, user, msg
                )
            )


    def _send(self, msg: str):
        if self._sock:
            self._sock.send(f"{msg}\r\n".encode())

    def _close_socket(self):
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None


# --------------------------------------------------
# 对外唯一接口
# --------------------------------------------------
_twitch_chat: Optional[SimpleTwitchChat] = None


async def start_twitch_task(config: dict, on_msg_cb: Callable[[str, str, str], None]):
    global _twitch_chat
    if _twitch_chat:
        return
    token = config.get("twitch_access_token", "")
    channel = config.get("twitch_channel", "")
    if not (token and channel):
        raise ValueError("Twitch token 或频道为空")

    _twitch_chat = SimpleTwitchChat(token, channel)
    _twitch_chat.set_callback(on_msg_cb)
    await _twitch_chat.start()
    print("[Twitch] 监听任务已启动")


async def stop_twitch_task():
    global _twitch_chat
    if _twitch_chat:
        await _twitch_chat.stop()
        _twitch_chat = None
        print("[Twitch] 监听任务已停止")
