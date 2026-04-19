# feishu_bot_manager.py
import asyncio
import json
import random
import threading
from typing import Optional, List
import weakref
import aiohttp
import io
import base64
import logging
import re
from pydantic import BaseModel, Field
from openai import AsyncOpenAI



from py.get_setting import convert_to_opus_simple, get_port, load_settings

from py.behavior_engine import BehaviorItem, global_behavior_engine, BehaviorSettings
from py.random_topic import get_random_topics
# 飞书机器人配置模型
class FeishuBotConfig(BaseModel):
    FeishuAgent: str          # LLM模型名
    memoryLimit: int          # 记忆条数限制
    appid: str                # 飞书APP_ID
    secret: str               # 飞书APP_SECRET
    separators: List[str]     # 消息分段符
    reasoningVisible: bool    # 是否显示推理过程
    quickRestart: bool        # 快速重启指令开关
    enableTTS: bool         # 是否启用TTS
    wakeWord: str              # 唤醒词
    # 行为规则设置 (与前端共用的结构)
    behaviorSettings: Optional[BehaviorSettings] = None
    # 飞书特定的推送目标ID列表 (配置一次，永久有效)
    behaviorTargetChatIds: List[str] = Field(default_factory=list)

class FeishuBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_client: Optional[FeishuClient] = None
        self.is_running = False
        self.config = None
        self.loop = None
        self._shutdown_event = threading.Event()
        self._startup_complete = threading.Event()
        self._ready_complete = threading.Event()
        self._startup_error = None
        self.ws = None  # 飞书长连接客户端
        self._stop_requested = False  # 添加停止请求标志
        
    def start_bot(self, config):
        """在新线程中启动飞书机器人"""
        if self.is_running:
            raise Exception("飞书机器人已在运行")
            
        self.config = config
        self._shutdown_event.clear()
        self._startup_complete.clear()
        self._ready_complete.clear()
        self._startup_error = None
        self._stop_requested = False
        
        # 使用线程方式启动
        self.bot_thread = threading.Thread(
            target=self._run_bot_thread,
            args=(config,),
            daemon=True,
            name="FeishuBotThread"
        )
        self.bot_thread.start()
        
        # 等待启动确认
        if not self._startup_complete.wait(timeout=30):
            self.stop_bot()
            raise Exception("飞书机器人连接超时")
            
        # 检查是否有启动错误
        if self._startup_error:
            self.stop_bot()
            raise Exception(f"飞书机器人启动失败: {self._startup_error}")
        
        # 等待机器人就绪
        if not self._ready_complete.wait(timeout=30):
            self.stop_bot()
            raise Exception("飞书机器人就绪超时，请检查网络连接和配置")
            
        if not self.is_running:
            self.stop_bot()
            raise Exception("飞书机器人未能正常运行")
    
    def _run_bot_thread(self, config):
        """线程中运行飞书机器人"""
        self.loop = None
        
        try:
            # 创建新的事件循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # --- 1. 创建飞书客户端 ---
            self.bot_client = FeishuClient()
            self.bot_client.FeishuAgent = config.FeishuAgent
            self.bot_client.memoryLimit = config.memoryLimit
            self.bot_client.separators = config.separators if config.separators else ['。', '\n', '？', '！']
            self.bot_client.reasoningVisible = config.reasoningVisible
            self.bot_client.quickRestart = config.quickRestart
            self.bot_client.appid = config.appid
            self.bot_client.secret = config.secret
            self.bot_client.enableTTS = config.enableTTS
            self.bot_client.wakeWord = config.wakeWord
            
            # 设置弱引用和回调
            self.bot_client._manager_ref = weakref.ref(self)
            self.bot_client._ready_callback = self._on_bot_ready

            # --- 2. 关键修复：强制同步最新的行为配置 ---
            # 即使传入的 config 不完整，这里也会重新加载全局设置来补全
            try:
                # 这是一个同步调用，在线程开始时运行是安全的
                settings = asyncio.run(load_settings())
                
                # 获取行为设置
                behavior_data = settings.get("behaviorSettings", {})
                
                # 获取飞书特定的目标列表 (可能在 config 里，也可能在 feishuBotConfig 里)
                # 优先用 config 里的，如果没有则去 settings 找
                target_ids = config.behaviorTargetChatIds
                if not target_ids:
                    feishu_conf = settings.get("feishuBotConfig", {})
                    target_ids = feishu_conf.get("behaviorTargetChatIds", [])
                
                # 构造更新数据
                if behavior_data:
                    logging.info(f"飞书线程: 检测到行为配置，正在同步... 目标群组数: {len(target_ids)}")
                    target_map = {"feishu": target_ids}
                    # 更新全局引擎
                    global_behavior_engine.update_config(behavior_data, target_map)
                    # 更新本地 config 对象以保持一致
                    config.behaviorSettings = behavior_data if isinstance(behavior_data, BehaviorSettings) else BehaviorSettings(**behavior_data)
                    config.behaviorTargetChatIds = target_ids
                else:
                    logging.warning("飞书线程: 未找到行为配置 behaviorSettings")
            except Exception as e:
                logging.error(f"飞书线程同步行为配置失败: {e}")
                import traceback
                print(traceback.format_exc())
            import lark_oapi as lark
            # --- 3. 初始化飞书 SDK ---
            lark_client = lark.Client.builder()\
                .app_id(config.appid)\
                .app_secret(config.secret)\
                .log_level(lark.LogLevel.INFO)\
                .build()
                
            self.bot_client.lark_client = lark_client
            
            # 创建事件处理程序
            event_dispatcher = lark.EventDispatcherHandler.builder("", "")\
                .register_p2_im_message_receive_v1(self.bot_client.sync_handle_message)\
                .build()
                
            # 创建长连接
            self.ws = lark.ws.Client(
                config.appid, 
                config.secret,
                event_handler=event_dispatcher,
                log_level=lark.LogLevel.INFO,
                auto_reconnect=False
            )
            
            # 在事件循环中运行WebSocket客户端
            self.loop.run_until_complete(self._async_run_websocket())
            
        except Exception as e:
            if not self._stop_requested:
                print(f"飞书机器人线程异常: {e}")
                if not self._startup_error:
                    self._startup_error = str(e)
            # 确保外部等待能解除
            if not self._startup_complete.is_set():
                self._startup_complete.set()
            if not self._ready_complete.is_set():
                self._ready_complete.set()
        finally:
            self._cleanup()  

    async def _async_run_websocket(self):
        """异步运行WebSocket连接"""
        try:
            # 建立连接
            await self.ws._connect()
            
            # 设置启动完成标志
            self._startup_complete.set()
            self._ready_complete.set()
            self.is_running = True
            logging.info("飞书机器人WebSocket连接已建立")
            
            # 启动ping循环
            ping_task = asyncio.create_task(self.ws._ping_loop())
            
            # 启动消息接收循环
            receive_task = asyncio.create_task(self._message_receive_loop())
            
            # --- 修复行为引擎启动逻辑 ---
            # 1. 如果引擎声称在运行，但 loop 不一致，或者为了保险起见，先停止它
            if global_behavior_engine.is_running:
                logging.info("检测到行为引擎已在运行，正在重启以适配当前事件循环...")
                global_behavior_engine.stop()
                # 给一点时间让旧循环的 task 退出
                await asyncio.sleep(0.5)

            # 2. 在当前线程的 Loop 中启动引擎
            behavior_task = asyncio.create_task(global_behavior_engine.start())
            logging.info("行为引擎已在飞书线程启动")
            
            # 等待任务完成或停止信号
            tasks = [ping_task, receive_task, behavior_task]
                
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                logging.info("WebSocket任务被取消")
            except Exception as e:
                if not self._stop_requested:
                    print(f"WebSocket任务异常: {e}")
                    
        except Exception as e:
            if not self._stop_requested:
                print(f"WebSocket连接失败: {e}")
                self._startup_error = str(e)
            raise

    async def _message_receive_loop(self):
        """消息接收循环"""
        try:
            while not self._stop_requested and not self._shutdown_event.is_set():
                if self.ws._conn is None:
                    break
                    
                try:
                    # 设置超时接收消息
                    msg = await asyncio.wait_for(self.ws._conn.recv(), timeout=1.0)
                    # 处理消息
                    asyncio.create_task(self.ws._handle_message(msg))
                except asyncio.TimeoutError:
                    # 超时是正常的，继续循环
                    continue
                except Exception as e:
                    if not self._stop_requested:
                        print(f"接收消息异常: {e}")
                    break
                    
        except asyncio.CancelledError:
            logging.info("消息接收循环被取消")
        except Exception as e:
            if not self._stop_requested:
                print(f"消息接收循环异常: {e}")
    
    def _on_bot_ready(self):
        """机器人就绪回调"""
        self.is_running = True
        if not self._ready_complete.is_set():
            self._ready_complete.set()
        logging.info("飞书机器人已完全就绪")

    def _cleanup(self):
        """清理资源"""
        self.is_running = False
        logging.info("开始清理飞书机器人资源...")
        
        # 1. 停止行为引擎 (至关重要)
        try:
            if global_behavior_engine.is_running:
                global_behavior_engine.stop()
                logging.info("行为引擎已停止")
        except Exception as e:
            logging.warning(f"停止行为引擎失败: {e}")

        # 2. 关闭长连接
        if self.ws and self.loop and not self.loop.is_closed():
            try:
                if asyncio.iscoroutinefunction(self.ws._disconnect):
                    self.loop.run_until_complete(self.ws._disconnect())
                logging.info("飞书长连接已关闭")
            except Exception as e:
                logging.warning(f"关闭飞书长连接时出错: {e}")
        
        # 3. 清理事件循环
        if self.loop and not self.loop.is_closed():
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    if not task.done():
                        task.cancel()
                
                if pending:
                    try:
                        self.loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    except Exception as e:
                        pass
                
                self.loop.close()
                logging.info("事件循环已关闭")
            except Exception as e:
                logging.warning(f"关闭事件循环时出错: {e}")
                
        self.bot_client = None
        self.loop = None
        self.ws = None
        self._shutdown_event.set()
        logging.info("飞书机器人资源清理完成")    

    def stop_bot(self):
        """停止飞书机器人"""
        if not self.is_running and not self.bot_thread:
            logging.info("飞书机器人未在运行")
            return
            
        logging.info("正在停止飞书机器人...")
        
        # 设置停止标志
        self._stop_requested = True
        self._shutdown_event.set()
        self.is_running = False
        
        # 如果有事件循环，尝试优雅停止
        if self.loop and not self.loop.is_closed():
            try:
                # 获取所有任务并取消它们
                try:
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        if not task.done():
                            task.cancel()
                except RuntimeError:
                    pass  # 事件循环可能已经关闭
                    
                # 关闭WebSocket连接
                if self.ws and hasattr(self.ws, '_disconnect'):
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self.ws._disconnect(), 
                            self.loop
                        )
                        future.result(timeout=2)
                        logging.info("WebSocket连接已关闭")
                    except Exception as e:
                        logging.warning(f"关闭WebSocket连接时出错: {e}")
                        
            except Exception as e:
                logging.warning(f"优雅停止过程中出错: {e}")
        
        # 等待线程结束
        if self.bot_thread and self.bot_thread.is_alive():
            try:
                logging.info("等待飞书机器人线程结束...")
                self.bot_thread.join(timeout=5)
                if self.bot_thread.is_alive():
                    logging.warning("飞书机器人线程在5秒超时后仍在运行，但这是预期的清理行为")
                else:
                    logging.info("飞书机器人线程已正常结束")
            except Exception as e:
                logging.warning(f"等待线程结束时出错: {e}")
        
        # 重置停止标志
        self._stop_requested = False
        logging.info("飞书机器人停止操作完成")

    def get_status(self):
        """获取机器人状态"""
        return {
            "is_running": self.is_running,
            "thread_alive": self.bot_thread.is_alive() if self.bot_thread else False,
            "client_ready": self.bot_client._is_ready if self.bot_client else False,
            "config": self.config.model_dump() if self.config else None,
            "loop_running": self.loop and not self.loop.is_closed() if self.loop else False,
            "startup_error": self._startup_error,
            "connection_established": self._startup_complete.is_set(),
            "ready_completed": self._ready_complete.is_set(),
            "stop_requested": self._stop_requested
        }

    def __del__(self):
        """析构函数确保资源清理"""
        try:
            self.stop_bot()
        except:
            pass

    def update_behavior_config(self, config: FeishuBotConfig):
        """
        热更新行为配置，不重启机器人
        """
        # 更新 Manager 的本地记录
        self.config = config
        
        # 1. 更新 Client 内部的实时参数
        if self.bot_client:
            self.bot_client.FeishuAgent = config.FeishuAgent 
            self.bot_client.enableTTS = config.enableTTS
            self.bot_client.wakeWord = config.wakeWord

        # 2. 更新全局行为引擎
        # 构造平台目标映射
        target_map = {
            "feishu": config.behaviorTargetChatIds
        }
        
        # 调用引擎更新 (会自动重置计时器)
        global_behavior_engine.update_config(
            config.behaviorSettings,
            target_map
        )
        logging.info("飞书机器人: 行为配置已热更新，计时器已重置")


class FeishuClient:
    def __init__(self):
        self.FeishuAgent = "super-model"
        self.memoryLimit = 10
        self.memoryList = {}
        self.asyncToolsID = {}
        self.fileLinks = {}
        self.separators = ['。', '\n', '？', '！']
        self.reasoningVisible = False
        self.quickRestart = True
        self._is_ready = False
        self.appid = None
        self.secret = None
        self.lark_client = None
        self.port = get_port()
        self._shutdown_requested = False
        self._manager_ref = None
        self._ready_callback = None
        self.enableTTS = False
        self.wakeWord = None
        
        # --- 新增：注册到行为引擎 ---
        # 告知引擎：飞书平台的执行逻辑由我负责
        global_behavior_engine.register_handler("feishu", self.execute_behavior_event)
        
    def sync_handle_message(self, data) -> None:
        """同步消息处理函数，用于注册到飞书事件分发器"""
        # 检查是否已请求停止
        if self._shutdown_requested:
            return
            
        # 检查管理器是否请求停止
        if self._manager_ref:
            manager = self._manager_ref()
            if manager and manager._stop_requested:
                return
        
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # 如果当前线程没有事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 检查事件循环是否已关闭
            if loop.is_closed():
                return
            
            # 在事件循环中运行异步函数
            future = asyncio.run_coroutine_threadsafe(
                self.handle_message(data), 
                loop
            )
            
            # 可选：等待结果（如果需要）
            # future.result(timeout=30)
        except Exception as e:
            if not self._shutdown_requested:
                print(f"处理消息时发生异常: {e}")

    async def handle_message(self, data) -> None:
        """处理飞书消息的主函数"""
        # 1. 基础检查
        if self._shutdown_requested: return
        if self._manager_ref:
            manager = self._manager_ref()
            if manager and (manager._stop_requested or not manager.is_running): return
        
        # 标记就绪
        if not self._is_ready:
            self._is_ready = True
            if self._ready_callback: self._ready_callback()
        
        msg = data.event.message
        msg_type = msg.message_type
        chat_id = msg.chat_id
        
        # --- 新增：上报活跃状态到引擎，用于无输入检测 ---
        global_behavior_engine.report_activity("feishu", chat_id)
        
        logging.info(f"收到 {msg.chat_type} 消息，类型：{msg_type}")
        
        # 2. 初始化 API 客户端
        from py.get_setting import load_settings
        settings = await load_settings()
        client = AsyncOpenAI(
            api_key="super-secret-key",
            base_url=f"http://127.0.0.1:{self.port}/v1"
        )
        
        # 3. 初始化记忆列表
        if chat_id not in self.memoryList:
            self.memoryList[chat_id] = []
            
        # =========================================================
        # 第一阶段：解析用户消息
        # =========================================================
        user_content = []  # 多模态内容
        user_text = ""     # 纯文本内容
        has_image = False  # 标记
        
        # --- (A) 文本消息 ---
        if msg_type == "text":
            try:
                text = json.loads(msg.content).get("text", "")

                # [新增] /id 指令：获取当前会话 ID
                if "/id" in text.lower():
                    # 飞书的 chat_id (open_chat_id) 通用于单聊和群聊
                    info_msg = (
                        f"🤖 **会话信息识别成功**\n\n"
                        f"当前 ChatID:\n`{chat_id}`\n\n"
                        f"💡 说明: 无论是群聊还是单聊，请直接复制上方 ID 填入自主行为的目标列表。"
                    )
                    await self._send_text(msg, info_msg)
                    return

                # 处理重启命令
                if self.quickRestart and text and ("/重启" in text or "/restart" in text):
                    self.memoryList[chat_id] = []
                    await self._send_text(msg, "对话记录已重置。")
                    return
                user_text = text
                if self.wakeWord and self.wakeWord not in user_text:
                    logging.info(f"未检测到唤醒词: {self.wakeWord}")
                    return
            except Exception as e:
                print(f"文本解析失败：{e}")
                return

        # --- (B) 图片消息 ---
        elif msg_type == "image":
            try:
                image_key = json.loads(msg.content).get("image_key", "")
                if image_key:
                    from lark_oapi.api.im.v1 import GetMessageResourceRequest as ResReq
                    # 下载图片逻辑
                    res_req = ResReq.builder().message_id(msg.message_id).file_key(image_key).type("image").build()
                    res_resp = self.lark_client.im.v1.message_resource.get(res_req)
                    if res_resp.success():
                        img_bin = res_resp.file.read()
                        base64_data = base64.b64encode(img_bin).decode("utf-8")
                        has_image = True
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}})
                        # 如果有附带文本
                        if "text" in json.loads(msg.content):
                            user_text = json.loads(msg.content).get("text", "")
            except Exception as e:
                print(f"图片处理失败：{e}")

        # --- (C) 富文本消息 (Post) ---
        elif msg_type == "post":
            try:
                content_json = json.loads(msg.content)
                user_text = self._extract_text_from_post(content_json)
                image_keys = self._extract_images_from_post(content_json)
                for image_key in image_keys:
                    res_req = ResReq.builder().message_id(msg.message_id).file_key(image_key).type("image").build()
                    res_resp = self.lark_client.im.v1.message_resource.get(res_req)
                    if res_resp.success():
                        img_bin = res_resp.file.read()
                        base64_data = base64.b64encode(img_bin).decode("utf-8")
                        has_image = True
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}})
            except Exception as e:
                print(f"富文本处理失败: {e}")

        # --- (D) 音频消息 (Audio) ---
        elif msg_type == "audio":
            try:
                content_json = json.loads(msg.content)
                file_key = content_json.get("file_key", "")
                if file_key:
                    res_req = ResReq.builder().message_id(msg.message_id).file_key(file_key).type("file").build()
                    res_resp = self.lark_client.im.v1.message_resource.get(res_req)
                    if res_resp.success():
                        audio_data = res_resp.file.read()
                        # 语音转文字 (ASR)
                        transcribed_text = await self._transcribe_audio(audio_data, file_key)
                        if transcribed_text:
                            user_text = transcribed_text
                            if self.wakeWord and self.wakeWord not in user_text:
                                return
                        else:
                            await self._send_text(msg, "语音转文字失败")
                            return
            except Exception as e:
                print(f"音频处理失败：{e}")
        
        else:
            await self._send_text(msg, f"暂不支持的消息类型：{msg_type}")
            return

        # =========================================================
        # 第二阶段：将解析后的内容添加到记忆
        # =========================================================
        if has_image:
            if user_text:
                user_content.append({"type": "text", "text": user_text})
            if user_content:
                self.memoryList[chat_id].append({"role": "user", "content": user_content})
            else:
                return # 无有效内容
        else:
            if user_text:
                self.memoryList[chat_id].append({"role": "user", "content": user_text})
            else:
                logging.warning("未检测到有效内容，跳过")
                return

        # =========================================================
        # 第三阶段：调用 API 并处理响应
        # =========================================================
        state = {
            "text_buffer": "",
            "image_buffer": "",
            "image_cache": [],
            "audio_buffer": []  # 音频缓冲区
        }
        
        try:
            asyncToolsID = self.asyncToolsID.get(chat_id, [])
            fileLinks = self.fileLinks.get(chat_id, [])
            if chat_id not in self.asyncToolsID: self.asyncToolsID[chat_id] = []
            if chat_id not in self.fileLinks: self.fileLinks[chat_id] = []
            
            # 调用 API
            stream = await client.chat.completions.create(
                model=self.FeishuAgent,
                messages=self.memoryList[chat_id],
                stream=True,
                extra_body={
                    "asyncToolsID": asyncToolsID,
                    "fileLinks": fileLinks,
                    "is_app_bot": True,
                }
            )
            
            full_response = []
            async for chunk in stream:
                reasoning_content = ""
                
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    # [捕获音频]
                    if hasattr(delta, "audio") and delta.audio:
                        if "data" in delta.audio:
                            state["audio_buffer"].append(delta.audio["data"])

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                         reasoning_content = delta.reasoning_content
                         
                    # 处理 async_tool_id 和 tool_link
                    if hasattr(delta, "async_tool_id") and delta.async_tool_id:
                        tid = delta.async_tool_id
                        if tid not in self.asyncToolsID[chat_id]: self.asyncToolsID[chat_id].append(tid)
                        else: self.asyncToolsID[chat_id].remove(tid)
                    
                    if hasattr(delta, "tool_link") and delta.tool_link:
                        if settings["tools"]["toolMemorandum"]["enabled"]:
                            self.fileLinks[chat_id].append(delta.tool_link)

                # 获取内容
                content = chunk.choices[0].delta.content or ""
                full_response.append(content)
                
                if reasoning_content and self.reasoningVisible:
                    content = reasoning_content
                
                state["text_buffer"] += content
                state["image_buffer"] += content
                
                # 实时发送文本
                if state["text_buffer"]:
                    force_split = len(state["text_buffer"]) > 4000
                    while True:
                        buffer = state["text_buffer"]
                        split_pos = -1
                        in_code_block = False
                        if force_split:
                            min_idx = len(buffer) + 1
                            found_sep_len = 0
                            for sep in self.separators:
                                idx = buffer.find(sep)
                                if idx != -1 and idx < min_idx:
                                    min_idx = idx
                                    found_sep_len = len(sep)
                            if min_idx <= len(buffer): split_pos = min_idx + found_sep_len
                        else:
                            i = 0
                            while i < len(buffer):
                                if buffer[i:].startswith("```"):
                                    in_code_block = not in_code_block
                                    i += 3
                                    continue
                                if not in_code_block:
                                    found_sep = False
                                    for sep in self.separators:
                                        if buffer[i:].startswith(sep):
                                            split_pos = i + len(sep)
                                            found_sep = True
                                            break
                                    if found_sep: break
                                i += 1
                        if split_pos == -1: break
                        
                        current_chunk = buffer[:split_pos]
                        state["text_buffer"] = buffer[split_pos:]
                        
                        clean_text = self._clean_text(current_chunk)
                        if clean_text: await self._send_text(msg, clean_text)
                        if force_split: break
            
            # 处理剩余内容
            self._extract_images(state)
            if state["text_buffer"]:
                clean_text = self._clean_text(state["text_buffer"])
                if clean_text: await self._send_text(msg, clean_text)
            for img_url in state["image_cache"]:
                await self._send_image(img_url)
            
            # [核心] 处理 Omni 音频转码与发送
            has_omni_audio = False
            if state["audio_buffer"]:
                try:
                    full_audio_b64 = "".join(state["audio_buffer"])
                    raw_audio_bytes = base64.b64decode(full_audio_b64)
                    
                    # 异步转码 Opus
                    final_audio, is_opus = await asyncio.to_thread(
                        convert_to_opus_simple, 
                        raw_audio_bytes
                    )
                    await self._send_omni_response(msg, final_audio, is_opus)
                    has_omni_audio = True
                except Exception as e:
                    print(f"Omni 音频处理失败: {e}")

            # 更新记忆
            full_content = "".join(full_response)
            
            # 如果没生成 Omni 音频，且开启了旧 TTS，才用旧 TTS
            if self.enableTTS and not has_omni_audio:
                await self._send_voice(msg, full_content)
                
            self.memoryList[chat_id].append({"role": "assistant", "content": full_content})
            
            # 限制记忆
            if self.memoryLimit > 0:
                while len(self.memoryList[chat_id]) > self.memoryLimit * 2:
                    self.memoryList[chat_id].pop(0)
                    if self.memoryList[chat_id]: self.memoryList[chat_id].pop(0)
            
        except Exception as e:
            print(f"处理消息异常: {e}")
            await self._send_text(msg, f"机器人异常: {str(e)}")
    async def _send_omni_response(self, original_msg, audio_data: bytes, is_opus: bool):
        """发送 Omni 模型生成的音频 (支持语音气泡)"""
        try:
            file_obj = io.BytesIO(audio_data)
            
            if is_opus:
                # 转换成功：发送飞书语音消息 (Voice Bubble)
                file_type = "opus"
                file_name = "reply.opus"
                msg_type = "audio"
                logging.info("发送模式: 语音气泡 (Opus)")
            else:
                # 转换失败：降级为发送文件 (File Attachment)
                file_type = "wav" 
                file_name = "reply.wav"
                msg_type = "file"
                logging.info("发送模式: 普通文件 (Wav)")
            from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody
            # 1. 上传文件
            # 注意：飞书上传接口区分 file_type
            upload_req = CreateFileRequest.builder() \
                .request_body(
                    CreateFileRequestBody.builder()
                    .file_type(file_type) 
                    .file_name(file_name)
                    .file(file_obj)
                    .build()
                ).build()

            upload_resp = self.lark_client.im.v1.file.create(upload_req)
            
            if not upload_resp.success():
                print(f"音频上传失败: {upload_resp.code} - {upload_resp.msg}")
                return

            file_key = upload_resp.data.file_key

            # 2. 发送消息
            # 无论是 audio 还是 file，内容格式都是 {"file_key": "xxx"}
            content_str = json.dumps({"file_key": file_key})
            
            chat_type = original_msg.chat_type
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody, ReplyMessageRequest, ReplyMessageRequestBody
            # 构建请求对象
            if chat_type == "p2p":
                req_builder = CreateMessageRequest.builder() \
                    .receive_id_type("chat_id") \
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(original_msg.chat_id)
                        .msg_type(msg_type)
                        .content(content_str)
                        .build()
                    )
                resp = self.lark_client.im.v1.message.create(req_builder.build())
            else:
                req_builder = ReplyMessageRequest.builder() \
                    .message_id(original_msg.message_id) \
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type(msg_type)
                        .content(content_str)
                        .build()
                    )
                resp = self.lark_client.im.v1.message.reply(req_builder.build())

            if not resp.success():
                print(f"音频消息发送失败: {resp.code} - {resp.msg}")
            else:
                logging.info(f"音频发送成功，Message ID: {resp.data.message_id}")

        except Exception as e:
            print(f"发送Omni音频异常: {e}")
            import traceback
            print(traceback.format_exc())


    async def _transcribe_audio(self, audio_data: bytes, file_key: str) -> str:
        """调用本地ASR接口转换音频为文字"""
        try:
            # 准备音频文件
            audio_file = io.BytesIO(audio_data)
            
            # 根据file_key或其他信息推断音频格式，飞书通常使用ogg或m4a格式
            # 这里我们让ASR接口自动检测格式
            filename = f"{file_key}.ogg"  # 飞书音频通常是ogg格式
            
            # 准备multipart/form-data请求
            form_data = aiohttp.FormData()
            form_data.add_field('audio', 
                            audio_file, 
                            filename=filename, 
                            content_type='audio/ogg')
            form_data.add_field('format', 'auto')  # 让ASR自动检测格式
            
            # 调用本地ASR接口
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{self.port}/asr",
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=60)  # 设置超时时间
                ) as response:
                    
                    if response.status != 200:
                        print(f"ASR请求失败，状态码: {response.status}")
                        response_text = await response.text()
                        print(f"ASR错误响应: {response_text}")
                        return None
                    
                    # 解析响应
                    result = await response.json()
                    
                    if result.get("success", False):
                        transcribed_text = result.get("text", "").strip()
                        if transcribed_text:
                            logging.info(f"ASR识别成功，引擎: {result.get('engine', 'unknown')}, "
                                    f"格式: {result.get('format', 'unknown')}")
                            return transcribed_text
                        else:
                            logging.warning("ASR识别结果为空")
                            return None
                    else:
                        error_msg = result.get("error", "未知错误")
                        print(f"ASR识别失败: {error_msg}")
                        return None
                        
        except asyncio.TimeoutError:
            print("ASR请求超时")
            return None
        except Exception as e:
            print(f"ASR转换异常: {e}")
            import traceback
            print(traceback.format_exc())
            return None



    def clean_markdown(self, buffer):
        # Remove heading marks (#, ##, ### etc.)
        buffer = re.sub(r'#{1,6}\s', '', buffer, flags=re.MULTILINE)
        
        # Remove single Markdown formatting characters (*_~`) but keep if they appear consecutively
        buffer = re.sub(r'[*_~`]+', '', buffer)
        
        # Remove list item marks (- or * at line start)
        buffer = re.sub(r'^\s*[-*]\s', '', buffer, flags=re.MULTILINE)
        
        # Remove emoji and other Unicode symbols
        buffer = re.sub(r'[\u2600-\u27BF\u2700-\u27BF\U0001F300-\U0001F9FF]', '', buffer)
        
        # Remove Unicode surrogate pairs
        buffer = re.sub(r'[\uD800-\uDBFF][\uDC00-\uDFFF]', '', buffer)
        
        # Remove image marks (![alt](url))
        buffer = re.sub(r'!\[.*?\]\(.*?\)', '', buffer)
        
        # Remove link marks ([text](url)), keeping the text
        buffer = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', buffer)
        
        # Remove leading/trailing whitespace
        return buffer.strip()


    async def _send_voice(self, original_msg, text):
        """发送语音消息（opus专用版本）"""
        try:
            from py.get_setting import load_settings
            settings = await load_settings()
            tts_settings = settings.get("ttsSettings", {})
            index = 0
            text = self.clean_markdown(text)
            # 专门为飞书请求opus格式
            payload = {
                "text": text,
                "voice": "default",
                "ttsSettings": tts_settings,
                "index": index,
                "mobile_optimized": True,  # 飞书优化标志
                "format": "opus"           # 明确请求opus格式
            }

            logging.info(f"发送TTS请求（opus格式），文本长度: {len(text)}，引擎: {tts_settings.get('engine', 'edgetts')}")

            timeout = aiohttp.ClientTimeout(total=90, connect=30, sock_read=60)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"http://127.0.0.1:{self.port}/tts",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        print(f"TTS 请求失败: {resp.status}")
                        error_text = await resp.text()
                        print(f"TTS 错误响应: {error_text}")
                        await self._send_text(original_msg, "语音生成失败，请稍后重试")
                        return

                    opus_data = await resp.read()
                    audio_format = resp.headers.get("X-Audio-Format", "unknown")
                    
                    logging.info(f"TTS响应成功，opus大小: {len(opus_data) / 1024:.1f}KB，格式: {audio_format}")

                    if len(opus_data) < 100:
                        print(f"opus数据异常，大小仅 {len(opus_data)} 字节")
                        await self._send_text(original_msg, "语音生成异常，请重试")
                        return

                    # 检查文件大小（飞书限制）
                    max_size = 10 * 1024 * 1024  # 10MB
                    if len(opus_data) > max_size:
                        print(f"opus文件过大: {len(opus_data) / (1024*1024):.1f}MB")
                        await self._send_text(original_msg, "语音文件过大，请尝试较短的文本")
                        return

                    # 上传opus文件到飞书
                    opus_file = io.BytesIO(opus_data)
                    
                    logging.info("开始上传opus语音文件到飞书...")
                    from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody
                    try:
                        upload_req = CreateFileRequest.builder() \
                            .request_body(
                                CreateFileRequestBody.builder()
                                .file_type("opus")           # 飞书要求的opus类型
                                .file_name("voice.opus")     # opus文件名
                                .file(opus_file)
                                .build()
                            ).build()

                        upload_resp = self.lark_client.im.v1.file.create(upload_req)
                        
                    except Exception as upload_error:
                        print(f"构建opus上传请求失败: {upload_error}")
                        await self._send_text(original_msg, "语音上传失败，请重试")
                        return

                    # 检查上传结果
                    if not upload_resp.success():
                        print(f"上传opus语音失败: {upload_resp.code} - {upload_resp.msg}")
                        
                        # 详细的错误处理
                        if upload_resp.code == 234001:
                            await self._send_text(original_msg, "语音格式错误，请联系管理员")
                        elif upload_resp.code == 234002:
                            await self._send_text(original_msg, "语音文件过大，请尝试较短的文本")
                        elif upload_resp.code == 99991663:
                            await self._send_text(original_msg, "机器人权限不足，请检查应用权限")
                        else:
                            await self._send_text(original_msg, f"语音上传失败: {upload_resp.msg}")
                        return

                    file_key = upload_resp.data.file_key
                    logging.info(f"opus语音上传成功，file_key: {file_key}")

                    # 发送语音消息
                    chat_type = original_msg.chat_type
                    audio_content = json.dumps({"file_key": file_key})
                    from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody, ReplyMessageRequest, ReplyMessageRequestBody
                    try:
                        if chat_type == "p2p":
                            req = CreateMessageRequest.builder() \
                                .receive_id_type("chat_id") \
                                .request_body(
                                    CreateMessageRequestBody.builder()
                                    .receive_id(original_msg.chat_id)
                                    .msg_type("audio")
                                    .content(audio_content)
                                    .build()
                                ).build()
                            
                            send_resp = self.lark_client.im.v1.message.create(req)
                        else:
                            req = ReplyMessageRequest.builder() \
                                .message_id(original_msg.message_id) \
                                .request_body(
                                    ReplyMessageRequestBody.builder()
                                    .msg_type("audio")
                                    .content(audio_content)
                                    .build()
                                ).build()
                            
                            send_resp = self.lark_client.im.v1.message.reply(req)

                        if not send_resp.success():
                            print(f"发送opus语音消息失败: {send_resp.code} - {send_resp.msg}")
                            
                            if send_resp.code == 230002:
                                await self._send_text(original_msg, "语音消息格式不支持")
                            elif send_resp.code == 99991663:
                                await self._send_text(original_msg, "机器人无发送消息权限")
                            else:
                                await self._send_text(original_msg, f"语音发送失败: {send_resp.msg}")
                        else:
                            logging.info(f"opus语音消息发送成功，消息ID: {send_resp.data.message_id}")

                    except Exception as send_error:
                        print(f"发送opus语音消息异常: {send_error}")
                        await self._send_text(original_msg, "语音消息发送失败")

        except asyncio.TimeoutError:
            print("opus TTS请求超时")
            await self._send_text(original_msg, "语音生成超时，请稍后重试")
        except Exception as e:
            print(f"发送opus语音异常: {e}")
            import traceback
            print(traceback.format_exc())
            await self._send_text(original_msg, "语音功能暂时不可用，请稍后重试")


    # 修改 _extract_text_from_post 方法
    def _extract_text_from_post(self, post_content):
        """从富文本中提取文本内容"""
        extracted_text = []
        
        try:
            # 提取标题
            if isinstance(post_content, dict):
                title = post_content.get("title", "")
                if title:
                    extracted_text.append(title)
                
                # 提取内容
                if "content" in post_content and isinstance(post_content["content"], list):
                    for paragraph in post_content["content"]:
                        paragraph_text = []
                        
                        if isinstance(paragraph, list):
                            for element in paragraph:
                                if isinstance(element, dict) and "tag" in element:
                                    tag = element["tag"]
                                    
                                    # 处理文本元素
                                    if tag == "text" and "text" in element:
                                        paragraph_text.append(element["text"])
                                    
                                    # 处理超链接
                                    elif tag == "a" and "text" in element:
                                        paragraph_text.append(element.get("text", ""))
                                    
                                    # 处理@用户
                                    elif tag == "at":
                                        user_name = element.get("user_name", "")
                                        paragraph_text.append(f"@{user_name}")
                        
                        # 添加当前段落文本
                        if paragraph_text:
                            extracted_text.append(" ".join(paragraph_text))
                            
            # 打印提取结果的日志
            logging.info(f"提取的文本内容: {extracted_text}")
        except Exception as e:
            logging.warning(f"从富文本提取文本失败: {e}")
            import traceback
            print(traceback.format_exc())
        
        return "\n".join(extracted_text)

    # 修改 _extract_images_from_post 方法
    def _extract_images_from_post(self, post_content):
        """从富文本中提取图片key"""
        image_keys = []
        
        try:
            if isinstance(post_content, dict) and "content" in post_content:
                content_array = post_content["content"]
                
                if isinstance(content_array, list):
                    for paragraph in content_array:
                        if isinstance(paragraph, list):
                            for element in paragraph:
                                if isinstance(element, dict) and "tag" in element:
                                    tag = element["tag"]
                                    
                                    # 处理图片元素
                                    if tag == "img" and "image_key" in element:
                                        image_keys.append(element["image_key"])
                                        logging.info(f"找到图片key: {element['image_key']}")
                                    
                                    # 处理媒体元素
                                    elif tag == "media" and "image_key" in element:
                                        image_keys.append(element["image_key"])
                                        logging.info(f"找到媒体图片key: {element['image_key']}")
            
            logging.info(f"提取的图片keys: {image_keys}")
        except Exception as e:
            logging.warning(f"从富文本提取图片失败: {e}")
            import traceback
            print(traceback.format_exc())
        
        return image_keys



    
    def _extract_images(self, state):
        """从文本中提取图片链接"""
        buffer = state["image_buffer"]
        # 匹配Markdown图片格式
        pattern = r'!\[.*?\]\((https?://[^\s\)]+)'
        matches = re.finditer(pattern, buffer)
        for match in matches:
            state["image_cache"].append(match.group(1))
    
    def _clean_text(self, text: str) -> str:
        # 1. 移除 Markdown 图片 ![alt](url) -> 空
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # 移除html标签
        text = re.sub(r'<.*?>', '', text)
        return text.strip()
    
    async def _send_text(self, original_msg, text):
        """发送文本消息（使用富文本 Post 格式以支持 Markdown）"""
        print("发送文本消息", text)
        try:
            if not text:
                return
            
            # 构建富文本结构
            # 飞书 Post 结构: {"zh_cn": {"title": "可选标题", "content": [[Nodes]]}}
            # 我们使用 md 标签，它独占一个段落
            content_dict = {
                "zh_cn": {
                    "content": [
                        [
                            {
                                "tag": "md",
                                "text": text
                            }
                        ]
                    ]
                }
            }
            
            # 序列化为 JSON 字符串
            content_str = json.dumps(content_dict)
            
            chat_type = original_msg.chat_type
            msg_type = "post"  # 关键：改为 post 类型
            from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody, ReplyMessageRequest, ReplyMessageRequestBody
            if chat_type == "p2p":  # 私聊
                req = CreateMessageRequest.builder()\
                    .receive_id_type("chat_id")\
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(original_msg.chat_id)
                        .msg_type(msg_type)
                        .content(content_str)
                        .build()
                    ).build()
                
                resp = self.lark_client.im.v1.message.create(req)
                
            else:  # 群聊
                req = ReplyMessageRequest.builder()\
                    .message_id(original_msg.message_id)\
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type(msg_type)
                        .content(content_str)
                        .build()
                    ).build()
                
                resp = self.lark_client.im.v1.message.reply(req)
            
            if not resp.success():
                print(f"发送 Markdown 文本失败: {resp.code} {resp.msg}")
                # 如果发送失败（可能是 md 语法太复杂或有非法字符），可以考虑回退到纯文本
                # logging.info("尝试回退到纯文本发送...")
                # ... (可选的回退逻辑)
                
        except Exception as e:
            print(f"发送文本异常: {e}")
            import traceback
            print(traceback.format_exc())
                
    async def _send_image(self, original_msg, image_url):
        """发送图片消息"""
        try:
            # 下载图片
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        print(f"下载图片失败: {image_url}")
                        return
                    
                    image_data = await response.read()
            
            # 转换为文件对象
            img_file = io.BytesIO(image_data)
            from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody
            # 上传图片
            upload_req = CreateImageRequest.builder()\
                .request_body(
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(img_file)
                    .build()
                ).build()
            
            upload_resp = self.lark_client.im.v1.image.create(upload_req)
            
            if not upload_resp.success():
                print(f"上传图片失败: {upload_resp.msg}")
                return
            
            image_key = upload_resp.data.image_key
            
            # 发送图片消息
            chat_type = original_msg.chat_type
            from lark_oapi.api.im.v1 import  CreateMessageRequest, CreateMessageRequestBody, ReplyMessageRequest, ReplyMessageRequestBody
            if chat_type == "p2p":  # 私聊
                req = CreateMessageRequest.builder()\
                    .receive_id_type("chat_id")\
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(original_msg.chat_id)
                        .msg_type("image")
                        .content(json.dumps({"image_key": image_key}))
                        .build()
                    ).build()
                
                resp = self.lark_client.im.v1.message.create(req)
                
            else:  # 群聊
                req = ReplyMessageRequest.builder()\
                    .message_id(original_msg.message_id)\
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .msg_type("image")
                        .content(json.dumps({"image_key": image_key}))
                        .build()
                    ).build()
                
                resp = self.lark_client.im.v1.message.reply(req)
            
            if not resp.success():
                print(f"发送图片失败: {resp.code} {resp.msg}")
                
        except Exception as e:
            print(f"发送图片异常: {e}")

    async def execute_behavior_event(self, chat_id: str, behavior_item: BehaviorItem):
        """
        回调函数：响应行为引擎的指令
        """
        logging.info(f"[FeishuClient] 行为触发! 目标: {chat_id}, 动作类型: {behavior_item.action.type}")
        
        prompt_content = await self._resolve_behavior_prompt(behavior_item)
        if not prompt_content: return

        # 构造增强版 MockMessage，确保包含 _send_text 需要的所有属性
        class MockMessage:
            def __init__(self, cid):
                self.chat_id = cid
                self.message_id = None
                self.chat_type = "p2p" 

        mock_msg = MockMessage(chat_id)

        if chat_id not in self.memoryList:
            self.memoryList[chat_id] = []
        
        # 构造上下文
        messages = self.memoryList[chat_id].copy()
        messages.append({"role": "user", "content": f"[system]: {prompt_content}"})
        
        # 同时也同步到内存，否则 AI 回复后上下文会断层
        self.memoryList[chat_id].append({"role": "user", "content": f"[system]: {prompt_content}"})

        try:
            client = AsyncOpenAI(
                api_key="super-secret-key",
                base_url=f"http://127.0.0.1:{self.port}/v1"
            )
            
            response = await client.chat.completions.create(
                model=self.FeishuAgent,
                messages=messages,
                stream=False, 
                extra_body={
                    "is_app_bot": True,
                    "behavior_trigger": True
                }
            )
            
            reply_content = response.choices[0].message.content
            if reply_content:
                # 发送内容
                await self._send_text(mock_msg, reply_content)
                self.memoryList[chat_id].append({"role": "assistant", "content": reply_content})
                
                if self.enableTTS:
                    await self._send_voice(mock_msg, reply_content)
            
        except Exception as e:
            logging.error(f"[FeishuClient] 执行行为 API 调用失败: {e}")
    async def _resolve_behavior_prompt(self, behavior: BehaviorItem) -> str:
        """解析行为配置，生成具体的 Prompt 指令"""
        action = behavior.action
        
        if action.type == "prompt":
            return action.prompt
            
        elif action.type == "random":
            if not action.random or not action.random.events:
                return None
                
            events = action.random.events
            if action.random.type == "random":
                return random.choice(events)
            elif action.random.type == "order":
                idx = action.random.orderIndex
                if idx >= len(events):
                    idx = 0
                selected = events[idx]
                # 更新索引 (仅内存生效)
                action.random.orderIndex = idx + 1
                return selected
                
        return None            