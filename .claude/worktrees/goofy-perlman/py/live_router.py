"""
直播子路由：/api/live/*  +  /ws/live/danmu
功能与原来完全一致，prefix 写死在 router 里
"""
from __future__ import annotations
import asyncio, threading, http.cookies, aiohttp
import uuid
from typing import Optional, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import py.blivedm as blivedm
import py.blivedm.models.web as web_models
import py.blivedm.models.open_live as open_models
from py.ytdm import YouTubeDMClient
from py.twitch_service import start_twitch_task, stop_twitch_task
# ==========================  关键：一次写死前缀 ==========================
router = APIRouter(prefix="/api/live", tags=["live"])
# ====================================================================

# 全局变量存储直播客户端和相关状态
live_client = None
live_thread = None
current_loop = None
stop_event = None  # 新增：用于通知线程停止
yt_client: Optional[YouTubeDMClient] = None 
twitch_task = None
# Pydantic模型
class LiveConfig(BaseModel):
    bilibili_enabled: bool = False
    bilibili_type: str = "open_live"
    bilibili_room_id: str = ""
    bilibili_sessdata: str = ""
    bilibili_ACCESS_KEY_ID: str = ""
    bilibili_ACCESS_KEY_SECRET: str = ""
    bilibili_APP_ID: str = ""
    bilibili_ROOM_OWNER_AUTH_CODE: str = ""
    youtube_enabled: bool = False
    youtube_video_id: str = ""
    youtube_api_key: str = ""
    twitch_enabled: bool = False
    twitch_channel: str = ""
    twitch_access_token: str = ""

class LiveConfigRequest(BaseModel):
    config: LiveConfig

class ApiResponse(BaseModel):
    success: bool
    message: str

# WebSocket管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                disconnected.append(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# API路由
@router.post("/start", response_model=ApiResponse)
async def start_live(request: LiveConfigRequest):
    global live_client, live_thread, stop_event, yt_client, current_loop,twitch_task

    config = request.config

    # ① 主线程先缓存事件循环，供 YouTube 用
    current_loop = asyncio.get_running_loop()
    print('[Live] main loop cached ->', current_loop)
    try:
        
        if config.bilibili_enabled:
            if live_client is not None:
                return ApiResponse(success=False, message="直播监听已在运行")

            # if config.bilibili_type == "web":
            #     if not config.bilibili_room_id:
            #         return ApiResponse(success=False, message="请输入房间ID")
            # elif config.bilibili_type == "open_live":
            if not all([
                config.bilibili_ACCESS_KEY_ID,
                config.bilibili_ACCESS_KEY_SECRET,
                config.bilibili_APP_ID,
                config.bilibili_ROOM_OWNER_AUTH_CODE
            ]):
                return ApiResponse(success=False, message="请完整填写开放平台配置信息")
            
            # 创建停止事件
            stop_event = threading.Event()
            
            # 创建新线程运行直播监听
            live_thread = threading.Thread(target=run_live_client, args=(config.dict(),))
            live_thread.daemon = True
            live_thread.start()
            
        if config.youtube_enabled:
            if yt_client is not None:
                return ApiResponse(success=False, message="YouTube 监听已在运行")
            if not config.youtube_video_id or not config.youtube_api_key:
                return ApiResponse(success=False, message="请填写 YouTube videoId 与 API_KEY")

            def _yt_on_message(msg: dict):
                # 现在 current_loop 一定有值
                asyncio.run_coroutine_threadsafe(manager.broadcast(msg), current_loop)

            yt_client = YouTubeDMClient(
                api_key=config.youtube_api_key,
                video_id=config.youtube_video_id,
                on_message=_yt_on_message
            )
            yt_client.start()
        
        if config.twitch_enabled:
            if twitch_task is not None:
                return ApiResponse(success=False, message="Twitch 监听已在运行")
            if not (config.twitch_access_token and config.twitch_channel):
                return ApiResponse(success=False, message="请填写 Twitch token 与频道")

            async def _twitch_on_msg(chan, user, msg):
                await manager.broadcast({
                    'id': str(uuid.uuid4()),
                    "type": "message",
                    "content": f"{user} said: {msg}",
                    "danmu_type": "danmaku",
                    "platform": "twitch"
                })

            # 启动 Twitch 任务
            twitch_task = asyncio.create_task(
                start_twitch_task(config.dict(), _twitch_on_msg)
            )

        # 等待一下确保客户端启动
        await asyncio.sleep(0.5)
        
        return ApiResponse(success=True, message="直播监听启动成功")
    except Exception as e:
        return ApiResponse(success=False, message=f"启动失败: {str(e)}")

@router.post("/stop", response_model=ApiResponse)
async def stop_live():
    global live_client, live_thread, current_loop, stop_event, yt_client,twitch_task
    
    try:
        
        print("开始停止直播监听...")
        if live_client is not None:
            
            # 设置停止事件
            if stop_event:
                stop_event.set()
            
            # 如果有事件循环，在其中停止客户端
            if current_loop and not current_loop.is_closed():
                try:
                    # 创建一个任务来停止客户端
                    future = asyncio.run_coroutine_threadsafe(
                        stop_live_client(), 
                        current_loop
                    )
                    # 等待停止完成，最多等待5秒
                    future.result(timeout=5)
                    print("客户端停止成功")
                except asyncio.TimeoutError:
                    print("停止客户端超时")
                except Exception as e:
                    print(f"停止客户端时出错: {e}")
            
            # 等待线程结束
            if live_thread and live_thread.is_alive():
                live_thread.join(timeout=3)
                if live_thread.is_alive():
                    print("警告: 线程未能在超时时间内结束")

        if yt_client is not None:
            yt_client.stop()
            yt_client = None
            
        if twitch_task:
            await stop_twitch_task()
            twitch_task.cancel()
            try:
                await twitch_task
            except asyncio.CancelledError:
                pass
            twitch_task = None

        # 清理全局变量
        live_client = None
        live_thread = None
        stop_event = None
        current_loop = None

        print("直播监听停止完成")
        return ApiResponse(success=True, message="直播监听停止成功")
        
    except Exception as e:
        print(f"停止直播监听时出错: {e}")
        return ApiResponse(success=False, message=f"停止失败: {str(e)}")

async def stop_live_client():
    """停止直播客户端的异步函数"""
    global live_client
    
    if live_client:
        try:
            await live_client.stop_and_close()
            print("直播客户端已停止")
        except Exception as e:
            print(f"停止直播客户端时出错: {e}")
        finally:
            live_client = None

@router.post("/reload", response_model=ApiResponse)
async def reload_live(request: LiveConfigRequest):
    try:
        # 先停止
        stop_result = await stop_live()
        if not stop_result.success:
            return stop_result
            
        # 等待一下确保完全停止
        await asyncio.sleep(2)
        
        # 再启动
        return await start_live(request)
    except Exception as e:
        return ApiResponse(success=False, message=f"重载失败: {str(e)}")

@router.get("/status")
async def get_live_status():
    """获取当前直播监听服务的运行状态"""
    # 只要有任意一个平台的客户端在运行，就认为是在运行中
    is_running = (live_client is not None) or (yt_client is not None) or (twitch_task is not None)
    
    return {
        "is_running": is_running,
        "details": {
            "bilibili": live_client is not None,
            "youtube": yt_client is not None,
            "twitch": twitch_task is not None
        }
    }

# —————— WebSocket 路由 ——————
# 注意：WebSocket 想挂在 /ws/live/danmu，再新建一个 router 即可
ws_router = APIRouter(prefix="/ws/live", tags=["live"])

# WebSocket路由
@ws_router.websocket("/danmu")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接活跃，接收心跳消息
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def init_session(sessdata: str = "") -> Optional[aiohttp.ClientSession]:
    """初始化aiohttp会话"""
    cookies = http.cookies.SimpleCookie()
    if sessdata:
        cookies['SESSDATA'] = sessdata
        cookies['SESSDATA']['domain'] = 'bilibili.com'

    session = aiohttp.ClientSession()
    if sessdata:
        session.cookie_jar.update_cookies(cookies)
    return session

def run_live_client(config: dict):
    """在新线程中运行直播客户端"""
    global live_client, stop_event
    
    try:
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print("开始运行直播客户端...")
        
        # 运行异步函数
        loop.run_until_complete(start_live_client(config))
        
    except Exception as e:
        print(f"直播客户端运行错误: {e}")
        # 通知前端错误
        if loop and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(manager.broadcast({
                    'type': 'error',
                    'message': str(e)
                }), loop)
            except:
                pass
    finally:
        print("直播客户端线程结束")
        # 清理
        if loop and not loop.is_closed():
            try:
                loop.close()
            except:
                pass
        loop = None
        live_client = None

async def start_live_client(config: dict):
    """启动直播客户端"""
    global live_client, stop_event
    
    session = None
    
    try:
        bilibili_type = config.get('bilibili_type', 'open_live')
        
        # if bilibili_type == 'web':
        #     # Web类型客户端
        #     room_id = int(config.get('bilibili_room_id', 0))
        #     sessdata = config.get('bilibili_sessdata', '')
            
        #     # 初始化session
        #     session = init_session(sessdata)
            
        #     live_client = blivedm.BLiveClient(room_id, session=session)
        #     handler = WebSocketHandler()
        #     live_client.set_handler(handler)
            
        # elif bilibili_type == 'open_live':
        # 开放平台类型客户端
        access_key_id = config.get('bilibili_ACCESS_KEY_ID', '')
        access_key_secret = config.get('bilibili_ACCESS_KEY_SECRET', '')
        app_id = int(config.get('bilibili_APP_ID', 0))
        room_owner_auth_code = config.get('bilibili_ROOM_OWNER_AUTH_CODE', '')
        
        live_client = blivedm.OpenLiveClient(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            app_id=app_id,
            room_owner_auth_code=room_owner_auth_code,
        )
        handler = OpenLiveWebSocketHandler()
        live_client.set_handler(handler)
        
        # else:
        #     raise ValueError(f"不支持的直播类型: {bilibili_type}")
        
        # print(f"启动{bilibili_type}类型的直播客户端")
        live_client.start()
        
        # 保持运行，直到收到停止信号
        try:
            while not (stop_event and stop_event.is_set()):
                await asyncio.sleep(1)
            print("收到停止信号，准备停止客户端")
        except asyncio.CancelledError:
            print("客户端被取消")
            
    except Exception as e:
        print(f"启动直播客户端错误: {e}")
        raise
    finally:
        # 清理资源
        if live_client:
            try:
                await live_client.stop_and_close()
                print("客户端已关闭")
            except Exception as e:
                print(f"关闭客户端时出错: {e}")
        
        if session:
            try:
                await session.close()
                print("Session已关闭")
            except Exception as e:
                print(f"关闭Session时出错: {e}")

class WebSocketHandler(blivedm.BaseHandler):
    """Web类型WebSocket处理器"""
    
    def _on_heartbeat(self, client: blivedm.BLiveClient, message: web_models.HeartbeatMessage):
        print(f'[{client.room_id}] 心跳')

    def _on_danmaku(self, client: blivedm.BLiveClient, message: web_models.DanmakuMessage):
        msg_text = f'{message.uname}说：{message.msg}'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "danmaku"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))
    
    def _on_gift(self, client: blivedm.BLiveClient, message: web_models.GiftMessage):
        msg_text = f'{message.uname} 赠送{message.gift_name}x{message.num} （{message.coin_type}瓜子x{message.total_coin}）'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "gift"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))
    
    def _on_buy_guard(self, client: blivedm.BLiveClient, message: web_models.GuardBuyMessage):
        msg_text = f'{message.username} 上舰，大航海等级={message.guard_level}'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "buy_guard"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))
    
    def _on_super_chat(self, client: blivedm.BLiveClient, message: web_models.SuperChatMessage):
        msg_text = f'{message.uname}发送醒目留言：{message.message}'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "super_chat"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

    def _on_interact_word(self, client: blivedm.BLiveClient, message: web_models.InteractWordMessage):
        if message.msg_type == 1:
            msg_text =  f'{message.username} 进入房间'
            data = {
                'id': str(uuid.uuid4()),
                'type': 'message',
                'content': msg_text,
                "danmu_type": "enter_room"
            }
            print(msg_text)
            asyncio.create_task(manager.broadcast(data))
        elif message.msg_type == 2:
            msg_text = f'{message.username} 关注了你'
            data = {
                'id': str(uuid.uuid4()),
                'type': 'message',
                'content': msg_text,
                "danmu_type": "follow"
            }
            print(msg_text)
            asyncio.create_task(manager.broadcast(data))


class OpenLiveWebSocketHandler(blivedm.BaseHandler):
    """开放平台类型WebSocket处理器"""
    
    def _on_heartbeat(self, client: blivedm.OpenLiveClient, message: web_models.HeartbeatMessage):
        print(f'[开放平台] 心跳')

    def _on_open_live_danmaku(self, client: blivedm.OpenLiveClient, message: open_models.DanmakuMessage):
        msg_text = f'{message.uname}说：{message.msg}'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "danmaku"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

    def _on_open_live_gift(self, client: blivedm.OpenLiveClient, message: open_models.GiftMessage):
        coin_type = '金瓜子' if message.paid else '银瓜子'
        total_coin = message.price * message.gift_num
        msg_text = f'{message.uname} 赠送{message.gift_name}x{message.gift_num} （{coin_type}x{total_coin}）'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "gift"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

    def _on_open_live_buy_guard(self, client: blivedm.OpenLiveClient, message: open_models.GuardBuyMessage):
        msg_text = f'{message.user_info.uname} 上舰，大航海等级={message.guard_level}'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "buy_guard"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

    def _on_open_live_super_chat(self, client: blivedm.OpenLiveClient, message: open_models.SuperChatMessage):
        msg_text = f'{message.uname}发送醒目留言：{message.message}'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "super_chat"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

    def _on_open_live_like(self, client: blivedm.OpenLiveClient, message: open_models.LikeMessage):
        msg_text = f'{message.uname} 点赞'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "like"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

    def _on_open_live_enter_room(self, client: blivedm.OpenLiveClient, message: open_models.RoomEnterMessage):
        msg_text = f'{message.uname} 进入房间'
        data = {
            'id': str(uuid.uuid4()),
            'type': 'message',
            'content': msg_text,
            "danmu_type": "enter_room"
        }
        print(msg_text)
        asyncio.create_task(manager.broadcast(data))

# 导出两个 router，主文件分别 include 即可
__all__ = ["router", "ws_router"]