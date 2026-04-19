# py/ytdm.py
from __future__ import annotations
import asyncio, threading, time
import uuid
from googleapiclient.discovery import build
from typing import Optional, Callable

class YouTubeDMClient:
    """
    极简轮询客户端
    用法：client = YouTubeDMClient(api_key, video_id, on_message)
          client.start()   # 非阻塞，内部启动线程
          ...
          client.stop()    # 线程安全退出
    """
    def __init__(self,
                 api_key: str,
                 video_id: str,
                 on_message: Callable[[dict], None],
                 poll_interval: int = 5):
        self.api_key = api_key
        self.video_id = video_id
        self.on_message = on_message
        self.poll_interval = poll_interval

        self._yt = build("youtube", "v3", developerKey=api_key)
        self._chat_id: Optional[str] = None
        self._page_token: Optional[str] = None
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --------- 外部调用 ---------
    def start(self):
        """非阻塞启动"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """线程安全停止"""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval + 1)

    # --------- 内部轮询 ---------
    def _run(self):
        self._chat_id = self._get_live_chat_id()
        print('[YouTube] got chat_id:', self._chat_id)   # ← 新增
        if not self._chat_id:
            print('[YouTube] 未开播，线程退出')
            return

        while not self._stop_evt.is_set():
            try:
                self._poll_once()
                print('[YouTube] poll_once done')        # ← 新增
            except Exception as e:
                print('[YouTube] poll error:', e)
            time.sleep(self.poll_interval)

    def _get_live_chat_id(self) -> Optional[str]:
        rsp = self._yt.videos().list(
            id=self.video_id,
            part="liveStreamingDetails"
        ).execute()
        if not rsp["items"]:
            return None
        return rsp["items"][0]["liveStreamingDetails"].get("activeLiveChatId")

    def _poll_once(self):
        rsp = self._yt.liveChatMessages().list(
            liveChatId=self._chat_id,
            part="snippet,authorDetails",
            pageToken=self._page_token,
            maxResults=2000
        ).execute()

        for item in rsp["items"]:
            author = item["authorDetails"]["displayName"]
            text   = item["snippet"]["displayMessage"]
            msg = {
                'id': str(uuid.uuid4()),
                "type": "message",
                "content": f"{author} said: {text}",
                "danmu_type": "danmaku",
                "platform": "youtube"
            }
            # 关键：把消息抛给回调
            self.on_message(msg)

        self._page_token = rsp.get("nextPageToken")