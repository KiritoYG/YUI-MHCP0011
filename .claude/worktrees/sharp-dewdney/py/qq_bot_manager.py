# qq_bot_manager.py
import asyncio
import json
import threading
import os
from typing import Optional,List
import weakref
import aiohttp
import botpy
from botpy.message import C2CMessage, GroupMessage
from openai import AsyncOpenAI
import logging
import re
import time
from pydantic import BaseModel
import requests
from PIL import Image
import io
import base64
from py.get_setting import get_port,load_settings
from py.image_host import upload_image_host

# 定义请求体
class QQBotConfig(BaseModel):
    QQAgent: str
    memoryLimit: int
    appid: str
    secret: str
    separators: List[str]
    reasoningVisible: bool
    quickRestart: bool
    is_sandbox: bool

class QQBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_client: Optional[MyClient] = None
        self.is_running = False
        self.config = None
        self.loop = None
        self._shutdown_event = threading.Event()
        self._startup_complete = threading.Event()
        self._ready_complete = threading.Event()  # 新增：等待 on_ready 完成
        self._startup_error = None
        
    def start_bot(self, config):
        """在新线程中启动机器人"""
        if self.is_running:
            raise Exception("机器人已在运行")
            
        self.config = config
        self._shutdown_event.clear()
        self._startup_complete.clear()
        self._ready_complete.clear()  # 重置就绪状态
        self._startup_error = None
        
        # 使用传统线程方式，更稳定
        self.bot_thread = threading.Thread(
            target=self._run_bot_thread,
            args=(config,),
            daemon=True,
            name="QQBotThread"
        )
        self.bot_thread.start()
        
        # 等待启动确认（连接建立）
        if not self._startup_complete.wait(timeout=30):
            self.stop_bot()
            raise Exception("机器人连接超时")
            
        # 检查是否有启动错误
        if self._startup_error:
            self.stop_bot()
            raise Exception(f"机器人启动失败: {self._startup_error}")
        
        # 等待机器人就绪（on_ready 触发）
        if not self._ready_complete.wait(timeout=30):
            self.stop_bot()
            raise Exception("机器人就绪超时，请检查网络连接和配置")
            
        if not self.is_running:
            self.stop_bot()
            raise Exception("机器人未能正常运行")
            
    def _run_bot_thread(self, config):
        """在线程中运行机器人"""
        self.loop = None
        bot_task = None
        
        try:
            # 创建新的事件循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # 创建机器人客户端
            self.bot_client = MyClient(intents=botpy.Intents(public_messages=True),is_sandbox=config.is_sandbox)
            self.bot_client.QQAgent = config.QQAgent
            self.bot_client.memoryLimit = config.memoryLimit
            self.bot_client.separators = config.separators if config.separators else ['。', '\n', '？', '！']
            self.bot_client.reasoningVisible = config.reasoningVisible
            self.bot_client.quickRestart = config.quickRestart
            
            # 设置弱引用以避免循环引用
            self.bot_client._manager_ref = weakref.ref(self)
            # 设置就绪回调
            self.bot_client._ready_callback = self._on_bot_ready
            
            # 创建启动任务
            async def run_bot():
                try:
                    logging.info("开始连接QQ机器人...")
                    
                    # 启动机器人连接
                    await self.bot_client.start(appid=config.appid, secret=config.secret)
                    
                except asyncio.CancelledError:
                    logging.info("机器人任务被取消")
                except Exception as e:
                    logging.error(f"机器人运行时异常: {e}")
                    # 保存启动错误
                    self._startup_error = str(e)
                    # 确保启动等待被解除
                    if not self._startup_complete.is_set():
                        self._startup_complete.set()
                    raise
            
            # 创建并运行机器人任务
            bot_task = self.loop.create_task(run_bot())
            
            # 在连接建立后设置启动完成标志（但还未就绪）
            def connection_established():
                if not self._startup_error:
                    self._startup_complete.set()
                    logging.info("机器人连接已建立，等待就绪...")
            
            # 稍微延迟设置连接状态，让 start 方法有机会检测错误
            async def delayed_connection_check():
                await asyncio.sleep(2)  # 给连接2秒时间
                if not bot_task.done() and not self._startup_error:
                    connection_established()
            
            # 创建延迟检查任务
            check_task = self.loop.create_task(delayed_connection_check())
            
            # 运行主任务
            self.loop.run_until_complete(bot_task)
            
        except Exception as e:
            logging.error(f"机器人线程异常: {e}")
            # 确保错误被记录并传递
            if not self._startup_error:
                self._startup_error = str(e)
        finally:
            # 确保启动等待被解除
            if not self._startup_complete.is_set():
                self._startup_complete.set()
            if not self._ready_complete.is_set():
                self._ready_complete.set()
                
            # 确保任务被正确取消
            if bot_task and not bot_task.done():
                bot_task.cancel()
                try:
                    self.loop.run_until_complete(bot_task)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logging.warning(f"取消机器人任务时出错: {e}")
            
            self._cleanup()
    
    def _on_bot_ready(self):
        """机器人就绪回调"""
        self.is_running = True
        self._ready_complete.set()
        logging.info("QQ机器人已完全就绪")

    def _cleanup(self):
        """清理资源"""
        self.is_running = False
        
        # 清理机器人客户端
        if self.bot_client and self.loop and not self.loop.is_closed():
            try:
                # 标记客户端为关闭状态
                self.bot_client._shutdown_requested = True
                
                # 如果客户端有close方法且事件循环还在运行，尝试关闭
                if hasattr(self.bot_client, 'close'):
                    # 创建关闭任务并运行
                    async def close_client():
                        try:
                            await self.bot_client.close()
                        except Exception as e:
                            logging.warning(f"关闭客户端时出错: {e}")
                    
                    close_task = self.loop.create_task(close_client())
                    try:
                        self.loop.run_until_complete(close_task)
                    except Exception as e:
                        logging.warning(f"执行关闭任务时出错: {e}")
                        
            except Exception as e:
                logging.warning(f"清理机器人客户端时出错: {e}")
                
        # 清理事件循环
        if self.loop and not self.loop.is_closed():
            try:
                # 获取所有待执行的任务
                pending_tasks = []
                try:
                    pending_tasks = asyncio.all_tasks(self.loop)
                except RuntimeError:
                    # 如果事件循环已经停止，all_tasks 可能会抛出 RuntimeError
                    pass
                
                # 取消所有待执行的任务
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                
                # 如果有待处理的任务，等待它们完成或取消
                if pending_tasks:
                    try:
                        # 使用 gather 收集所有任务的结果
                        async def cancel_all_tasks():
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                        
                        cancel_task = self.loop.create_task(cancel_all_tasks())
                        self.loop.run_until_complete(cancel_task)
                        
                    except Exception as e:
                        logging.warning(f"等待任务取消时出错: {e}")
                        
                # 关闭事件循环
                if not self.loop.is_closed():
                    self.loop.close()
                        
            except Exception as e:
                logging.warning(f"关闭事件循环时出错: {e}")
                
        self.bot_client = None
        self.loop = None
        self._shutdown_event.set()
            
    def stop_bot(self):
        """停止机器人"""
        if not self.is_running and not self.bot_thread:
            return
            
        logging.info("正在停止QQ机器人...")
        
        # 设置停止标志
        self._shutdown_event.set()
        self.is_running = False
        
        # 如果机器人客户端存在，标记为请求关闭
        if self.bot_client:
            self.bot_client._shutdown_requested = True
        
        # 如果事件循环存在且正在运行，尝试停止它
        if self.loop and not self.loop.is_closed():
            try:
                # 在循环中调度停止
                self.loop.call_soon_threadsafe(self.loop.stop)
            except RuntimeError as e:
                # 如果循环已经停止，会抛出 RuntimeError
                logging.debug(f"事件循环已停止: {e}")
            except Exception as e:
                logging.warning(f"停止事件循环时出错: {e}")
        
        # 等待线程结束
        if self.bot_thread and self.bot_thread.is_alive():
            try:
                self.bot_thread.join(timeout=10)
                if self.bot_thread.is_alive():
                    logging.warning("机器人线程在超时后仍在运行")
            except Exception as e:
                logging.warning(f"等待线程结束时出错: {e}")
                
        logging.info("QQ机器人已停止")


    def get_status(self):
        """获取机器人状态"""
        return {
            "is_running": self.is_running,
            "thread_alive": self.bot_thread.is_alive() if self.bot_thread else False,
            "client_ready": self.bot_client.is_running if self.bot_client else False,
            "config": self.config.model_dump() if self.config else None,
            "loop_running": self.loop and not self.loop.is_closed() if self.loop else False,
            "startup_error": self._startup_error,
            "connection_established": self._startup_complete.is_set(),
            "ready_completed": self._ready_complete.is_set()
        }


    def __del__(self):
        """析构函数确保资源清理"""
        try:
            self.stop_bot()
        except:
            pass


# MyClient 类的修改
class MyClient(botpy.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_running = False
        self.QQAgent = "super-model"
        self.memoryLimit = 10
        self.memoryList = {}
        self.asyncToolsID = {}
        self.fileLinks = {}
        self.separators = ['。', '\n', '？', '！']
        self.reasoningVisible = False
        self.quickRestart = True
        self._ready_event = asyncio.Event()
        self.port = get_port()
        self._shutdown_requested = False
        self._manager_ref = None  # 弱引用管理器
        
    async def start(self, appid, secret):
        """启动客户端"""
        try:
            await super().start(appid=appid, secret=secret)
        except Exception as e:
            logging.error(f"客户端启动失败: {e}")
            # 确保错误被传递到上层
            raise Exception(f"认证失败或配置错误: {e}")
    
    async def close(self):
        """关闭客户端"""
        self._shutdown_requested = True
        self.is_running = False
        try:
            # 调用父类的关闭方法
            await super().close()
        except Exception as e:
            logging.warning(f"关闭客户端时出错: {e}")
    
    async def on_ready(self):
        """机器人就绪事件"""
        if self._shutdown_requested:
            return
            
        self.is_running = True
        self._ready_event.set()
        
        # 调用管理器的就绪回调
        if self._ready_callback:
            self._ready_callback()
        
        logging.info("QQ机器人已就绪，可以接收消息")

    async def wait_for_ready(self, timeout=30):
        """等待机器人就绪"""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    async def on_c2c_message_create(self, message: C2CMessage):
        if not self.is_running:
            return
        settings = await load_settings()
        client = AsyncOpenAI(
            api_key="super-secret-key",
            base_url=f"http://127.0.0.1:{self.port}/v1"
        )
        
        user_content = []
        image_url_list = []
        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type.startswith("image/"):
                    image_url = attachment.url
                    image_url_list.append(image_url)
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                # 获取原始图像数据
                                image_data = await response.read()
                                
                                # 检查是否为支持的格式
                                content_type = attachment.content_type.lower()
                                if content_type not in ["image/png", "image/jpeg", "image/gif"]:
                                    try:
                                        # 转换为JPG格式
                                        img = Image.open(io.BytesIO(image_data))
                                        if img.mode in ("RGBA", "LA", "P"):
                                            img = img.convert("RGB")
                                        
                                        jpg_buffer = io.BytesIO()
                                        img.save(jpg_buffer, format="JPEG", quality=95)
                                        image_data = jpg_buffer.getvalue()
                                        content_type = "image/jpeg"
                                    except Exception as e:
                                        print(f"图像转换失败: {e}")
                                        continue
                                
                                # 转换为Base64
                                base64_data = base64.b64encode(image_data).decode("utf-8")
                                data_uri = f"data:{content_type};base64,{base64_data}"
                                
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_uri
                                    }
                                })
        
        if user_content:
            user_content.append({"type": "text", "text": message.content+"图片链接："+json.dumps(image_url_list)})
        else:
            user_content = message.content
            
        print(f"User content: {user_content}")

        c_id = message.author.user_openid
        if c_id not in self.memoryList:
            self.memoryList[c_id] = []
            
        # 初始化状态管理
        if not hasattr(self, 'msg_seq_counters'):
            self.msg_seq_counters = {}
        self.msg_seq_counters.setdefault(c_id, 1)
        if not hasattr(self, 'processing_states'):
            self.processing_states = {}
        self.processing_states[c_id] = {
            "text_buffer": "",
            "image_buffer": "",
            "image_cache": []
        }

        if self.quickRestart:
            if "/重启" in message.content:
                self.memoryList[c_id] = []
                await self._send_text_message(message, "对话记录已重置。")
                return
            if "/restart" in message.content: 
                self.memoryList[c_id] = []
                await self._send_text_message(message, "The conversation record has been reset.")
                return

        self.memoryList[c_id].append({"role": "user", "content": user_content})

        try:
            asyncToolsID = []
            if c_id in self.asyncToolsID:
                asyncToolsID = self.asyncToolsID[c_id]
            else:
                self.asyncToolsID[c_id] = []
            if c_id in self.fileLinks:
                fileLinks = self.fileLinks[c_id]
            else:
                fileLinks = []
            # 流式调用API
            stream = await client.chat.completions.create(
                model=self.QQAgent,
                messages=self.memoryList[c_id],
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
                tool_content = ""
                if chunk.choices:
                    chunk_dict = chunk.model_dump()
                    delta = chunk_dict["choices"][0].get("delta", {})
                    if delta:
                        reasoning_content = delta.get("reasoning_content", "") 
                        tool_content = delta.get("tool_content", "")
                        async_tool_id = delta.get("async_tool_id", "")
                        tool_link = delta.get("tool_link", "")

                        if tool_link and settings["tools"]["toolMemorandum"]["enabled"]:
                            if c_id not in self.fileLinks:
                                self.fileLinks[c_id] = []
                            self.fileLinks[c_id].append(tool_link)

                        if async_tool_id:
                            # 判断async_tool_id在不在self.asyncToolsID[c_id]中
                            if async_tool_id not in self.asyncToolsID[c_id]:
                                self.asyncToolsID[c_id].append(async_tool_id)

                            # 如果async_tool_id在self.asyncToolsID[c_id]中，则删除
                            else:
                                self.asyncToolsID[c_id].remove(async_tool_id)

                content = chunk.choices[0].delta.content or ""
                full_response.append(content)
                if reasoning_content and self.reasoningVisible:
                    content = reasoning_content
                
                # 更新缓冲区
                state = self.processing_states[c_id]
                state["text_buffer"] += content
                state["image_buffer"] += content

                # 处理文本实时发送
                while True:
                    if self.separators == []:
                        break
                    # 查找分隔符
                    buffer = state["text_buffer"]
                    split_pos = -1
                    for i, c in enumerate(buffer):
                        if c in self.separators:
                            split_pos = i + 1
                            break
                    if split_pos == -1:
                        break

                    # 分割并处理当前段落
                    current_chunk = buffer[:split_pos]
                    state["text_buffer"] = buffer[split_pos:]
                    
                    # 清洗并发送文字
                    clean_text = self._clean_text(current_chunk)
                    if clean_text:
                        await self._send_text_message(message, clean_text)
                    
            # 提取图片到缓存
            self._extract_images_to_cache(c_id)

            # 处理剩余文本
            if state["text_buffer"]:
                clean_text = self._clean_text(state["text_buffer"])
                if clean_text:
                    await self._send_text_message(message, clean_text)
            
            # 最终图片发送
            await self._send_cached_images(message)

            # 记忆管理
            full_content = "".join(full_response)
            self.memoryList[c_id].append({"role": "assistant", "content": full_content})
            if self.memoryLimit > 0:
                while len(self.memoryList[c_id]) > self.memoryLimit:
                    self.memoryList[c_id].pop(0)

        except Exception as e:
            print(f"处理异常: {e}")
            clean_text = self._clean_text(str(e))
            if clean_text:
                await self._send_text_message(message, clean_text)
        finally:
            # 清理状态
            if c_id in self.processing_states:
                del self.processing_states[c_id]

    def _extract_images_to_cache(self, c_id):
        """渐进式图片链接提取"""
        state = self.processing_states[c_id]
        temp_buffer = state["image_buffer"]
        state["image_buffer"] = ""  # 重置缓冲区
        
        # 匹配完整图片链接
        pattern = r'!\[.*?\]\((https?://[^\s\)]+)'
        matches = re.finditer(pattern, temp_buffer)
        for match in matches:
            state["image_cache"].append(match.group(1))

    async def _send_text_message(self, message, text):
        """发送文本消息并更新序号"""
        c_id = message.author.user_openid
        await message._api.post_c2c_message(
            openid=message.author.user_openid,
            msg_type=0,
            msg_id=message.id,
            content=text,
            msg_seq=self.msg_seq_counters[c_id]
        )
        self.msg_seq_counters[c_id] += 1

    async def _send_cached_images(self, message):
        """批量发送缓存的图片"""
        c_id = message.author.user_openid
        state = self.processing_states.get(c_id, {})
        
        for url in state.get("image_cache", []):
            try:
                # 链接有效性验证
                if not re.match(r'^https?://', url):
                    continue
                # 判断是否开启了图床功能
                url = await upload_image_host(url)
                # 用request获取图片，保证图片存在
                res = requests.get(url)

                print(f"发送图片: {url}")
                # 上传媒体文件
                upload_media = await message._api.post_c2c_file(
                    openid=message.author.user_openid,
                    file_type=1,
                    url=url
                )
                # 发送富媒体消息
                await message._api.post_c2c_message(
                    openid=message.author.user_openid,
                    msg_type=7,
                    msg_id=message.id,
                    media=upload_media,
                    msg_seq=self.msg_seq_counters[c_id]
                )
                self.msg_seq_counters[c_id] += 1
            except Exception as e:
                print(f"图片发送失败: {e}")
                clean_text = self._clean_text(str(e))
                if clean_text:
                    await self._send_text_message(message, clean_text)

    def _clean_text(self, text):
        """三级内容清洗"""
        # 移除图片标记
        clean = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 移除超链接
        clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)
        # 移除纯URL
        clean = re.sub(r'https?://\S+', '', clean)
        return clean.strip()

    
    async def on_group_at_message_create(self, message: GroupMessage):
        if not self.is_running:
            return
        settings = await load_settings()
        client = AsyncOpenAI(
            api_key="super-secret-key",
            base_url=f"http://127.0.0.1:{self.port}/v1"
        )
        user_content = []
        image_url_list = []
        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type.startswith("image/"):
                    image_url = attachment.url
                    image_url_list.append(image_url)
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as response:
                            if response.status == 200:
                                # 获取原始图像数据
                                image_data = await response.read()
                                
                                # 检查是否为支持的格式
                                content_type = attachment.content_type.lower()
                                if content_type not in ["image/png", "image/jpeg", "image/gif"]:
                                    try:
                                        # 转换为JPG格式
                                        img = Image.open(io.BytesIO(image_data))
                                        if img.mode in ("RGBA", "LA", "P"):
                                            img = img.convert("RGB")
                                        
                                        jpg_buffer = io.BytesIO()
                                        img.save(jpg_buffer, format="JPEG", quality=95)
                                        image_data = jpg_buffer.getvalue()
                                        content_type = "image/jpeg"
                                    except Exception as e:
                                        print(f"图像转换失败: {e}")
                                        continue
                                
                                # 转换为Base64
                                base64_data = base64.b64encode(image_data).decode("utf-8")
                                data_uri = f"data:{content_type};base64,{base64_data}"
                                
                                user_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_uri
                                    }
                                })
        if user_content:
            user_content.append({"type": "text", "text": message.content+"图片链接："+json.dumps(image_url_list)})
        else:
            user_content = message.content
        g_id = message.group_openid
        if g_id not in self.memoryList:
            self.memoryList[g_id] = []
        # 初始化群组状态
        if not hasattr(self, 'group_states'):
            self.group_states = {}
        self.group_states[g_id] = {
            "msg_seq": 1,
            "text_buffer": "",
            "image_buffer": "",
            "image_cache": []
        }
        state = self.group_states[g_id]
        if self.quickRestart:
            if "/重启" in message.content:
                self.memoryList[g_id] = []
                await self._send_group_text(message, "对话记录已重置。", state)
                return
            if "/restart" in message.content: 
                self.memoryList[g_id] = []
                await self._send_group_text(message, "The conversation record has been reset.", state)
                return
        self.memoryList[g_id].append({"role": "user", "content": user_content})

        try:
            asyncToolsID = []
            if g_id in self.asyncToolsID:
                asyncToolsID = self.asyncToolsID[g_id]
            else:
                self.asyncToolsID[g_id] = []
            if g_id in self.fileLinks:
                fileLinks = self.fileLinks[g_id]
            else:
                fileLinks = []
            # 流式API调用
            stream = await client.chat.completions.create(
                model=self.QQAgent,
                messages=self.memoryList[g_id],
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
                tool_content = ""
                if chunk.choices:
                    chunk_dict = chunk.model_dump()
                    delta = chunk_dict["choices"][0].get("delta", {})
                    if delta:
                        reasoning_content = delta.get("reasoning_content", "")
                        tool_content = delta.get("tool_content", "")
                        async_tool_id = delta.get("async_tool_id", "")
                        tool_link = delta.get("tool_link", "")
                        if tool_link and settings["tools"]["toolMemorandum"]["enabled"]:
                            if g_id not in self.fileLinks:
                                self.fileLinks[g_id] = []
                            self.fileLinks[g_id].append(tool_link)
                        if async_tool_id:
                            # 判断async_tool_id在不在self.asyncToolsID[g_id]中
                            if async_tool_id not in self.asyncToolsID[g_id]:
                                self.asyncToolsID[g_id].append(async_tool_id)

                            # 如果async_tool_id在self.asyncToolsID[g_id]中，则删除
                            else:
                                self.asyncToolsID[g_id].remove(async_tool_id)
                       
                content = chunk.choices[0].delta.content or ""
                full_response.append(content)
                if reasoning_content and self.reasoningVisible:
                    content = reasoning_content
                
                # 更新文本缓冲区
                state["text_buffer"] += content
                state["image_buffer"] += content

                # 处理文本分段
                while True:
                    if self.separators == []:
                        break
                    # 查找分隔符（。或\n）
                    buffer = state["text_buffer"]
                    split_pos = -1
                    for i, c in enumerate(buffer):
                        if c in self.separators:
                            split_pos = i + 1
                            break
                    if split_pos == -1:
                        break

                    # 处理当前段落
                    current_chunk = buffer[:split_pos]
                    state["text_buffer"] = buffer[split_pos:]
                    
                    # 清洗并发送文字
                    clean_text = self._clean_group_text(current_chunk)
                    if clean_text:
                        await self._send_group_text(message, clean_text, state)
                    
            # 提取图片到缓存
            self._cache_group_images(g_id)

            # 处理剩余文本
            if self.group_states[g_id]["text_buffer"]:
                clean_text = self._clean_group_text(self.group_states[g_id]["text_buffer"])
                if clean_text:
                    await self._send_group_text(message, clean_text, state)

            # 发送缓存图片
            await self._send_group_images(message, g_id)

            # 记忆管理
            full_content = "".join(full_response)
            self.memoryList[g_id].append({"role": "assistant", "content": full_content})
            if self.memoryLimit > 0:
                while len(self.memoryList[g_id]) > self.memoryLimit:
                    self.memoryList[g_id].pop(0)

        except Exception as e:
            print(f"群聊处理异常: {e}")
            clean_text = self._clean_group_text(str(e))
            if clean_text:
                await self._send_group_text(message, clean_text, state)
        finally:
            # 清理状态
            del self.group_states[g_id]

    def _cache_group_images(self, g_id):
        """渐进式图片缓存"""
        state = self.group_states[g_id]
        temp_buffer = state["image_buffer"]
        state["image_buffer"] = ""
        
        # 匹配完整图片链接
        pattern = r'!\[.*?\]\((https?://[^\s\)]+)'
        matches = re.finditer(pattern, temp_buffer)
        for match in matches:
            state["image_cache"].append(match.group(1))

    async def _send_group_text(self, message, text, state):
        """发送群聊文字消息"""
        await message._api.post_group_message(
            group_openid=message.group_openid,
            msg_type=0,
            msg_id=message.id,
            content=text,
            msg_seq=state["msg_seq"]
        )
        state["msg_seq"] += 1

    async def _send_group_images(self, message, g_id):
        """批量发送群聊图片"""
        state = self.group_states.get(g_id, {})
        for url in state.get("image_cache", []):
            try:
                # 链接有效性验证
                if not url.startswith(('http://', 'https://')):
                    continue
                # 判断是否开启了图床功能
                url = await upload_image_host(url)
                # 用request获取图片，保证图片存在
                res = requests.get(url)
                print(f"发送图片: {url}")
                # 上传群文件
                upload_media = await message._api.post_group_file(
                    group_openid=message.group_openid,
                    file_type=1,
                    url=url
                )
                # 发送群媒体消息
                await message._api.post_group_message(
                    group_openid=message.group_openid,
                    msg_type=7,
                    msg_id=message.id,
                    media=upload_media,
                    msg_seq=state["msg_seq"]
                )
                state["msg_seq"] += 1
            except Exception as e:
                print(f"群图片发送失败: {e}")
                clean_text = self._clean_group_text(str(e))
                if clean_text:
                    await self._send_group_text(message, clean_text, state)

    def _clean_group_text(self, text):
        """群聊文本三级清洗"""
        # 移除图片标记
        clean = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 移除超链接
        clean = re.sub(r'\[.*?\]\(.*?\)', '', clean)
        # 移除纯URL
        clean = re.sub(r'https?://\S+', '', clean)
        return clean.strip()