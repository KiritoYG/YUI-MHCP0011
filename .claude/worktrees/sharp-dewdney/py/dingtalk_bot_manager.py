import asyncio
import json
import random
import threading
import os
import time
import logging
import aiohttp
import re
import base64
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# 钉钉官方 SDK
import dingtalk_stream
from dingtalk_stream import AckMessage, ChatbotMessage

# 假设这两个函数在你的 py.get_setting 中定义
from py.behavior_engine import BehaviorItem, BehaviorSettings,global_behavior_engine
from py.get_setting import get_port, load_settings
from py.random_topic import get_random_topics

# 配置模型
class DingtalkBotConfig(BaseModel):
    DingtalkAgent: str
    memoryLimit: int
    appKey: str
    appSecret: str
    separators: List[str]
    reasoningVisible: bool
    quickRestart: bool
    enableTTS: bool 
    wakeWord: str
    behaviorSettings: Optional[BehaviorSettings] = None
    behaviorTargetChatIds: List[str] = Field(default_factory=list)

class DingtalkBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.config = None
        self._startup_error = None
        self.client = None
        
    def start_bot(self, config: DingtalkBotConfig):
        if self.is_running:
            raise Exception("钉钉机器人已在运行")
        self.config = config
        self._startup_error = None
        self.bot_thread = threading.Thread(target=self._run_bot_thread, args=(config,), daemon=True)
        self.bot_thread.start()
        self.is_running = True

    def _run_bot_thread(self, config):
        """线程中运行钉钉机器人：修复版"""
        async def main_loop():
            try:
                # 1. 初始化逻辑类
                self.bot_logic = DingtalkClientLogic(config)
                
                # 2. 强制同步最新的行为配置
                from py.get_setting import load_settings
                settings = await load_settings()
                behavior_data = settings.get("behaviorSettings", {})
                target_ids = config.behaviorTargetChatIds or []
                
                if behavior_data:
                    logging.info(f"[Dingtalk] 同步行为配置中... 目标数: {len(target_ids)}")
                    global_behavior_engine.update_config(behavior_data, {"dingtalk": target_ids})

                # 3. 初始化钉钉官方 SDK (使用异步模式)
                credential = dingtalk_stream.Credential(config.appKey, config.appSecret)
                # 注意：这里我们手动管理 Loop
                self.client = dingtalk_stream.DingTalkStreamClient(credential)
                
                handler = DingtalkInternalHandler(self.bot_logic)
                self.client.register_callback_handler(ChatbotMessage.TOPIC, handler)
                
                logging.info("[Dingtalk] 正在并发启动：行为引擎 + 钉钉长连接...")

                # 4. 【核心修复】用 gather 同时运行两个异步长任务
                # client.start() 是异步的，不会阻塞 loop
                await asyncio.gather(
                    global_behavior_engine.start(),
                    self.client.start()
                )
                
            except Exception as e:
                self._startup_error = str(e)
                logging.error(f"[Dingtalk] 异步循环异常: {e}")
            finally:
                self.is_running = False
                global_behavior_engine.stop()

        # 在子线程中启动全新的 asyncio 事件循环
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(main_loop())
        except Exception as e:
            logging.error(f"[Dingtalk] 线程退出: {e}")

    def stop_bot(self):
        if self.client:
            try: self.client.stop()
            except: pass
        self.is_running = False

    def get_status(self):
        return {
            "is_running": self.is_running,
            "has_error": self._startup_error is not None,
            "error_message": self._startup_error,
            "config_loaded": self.config is not None
        }

    def update_behavior_config(self, config: DingtalkBotConfig):
        """
        热更新行为配置，不重启机器人
        """
        # 更新 Manager 的本地记录
        self.config = config
        
        # 1. 更新 Logic 内部的实时参数
        if self.bot_logic:
            self.bot_logic.config = config

        # 2. 更新全局行为引擎
        target_map = {
            "dingtalk": config.behaviorTargetChatIds
        }
        
        # 调用引擎更新 (会自动重置计时器)
        global_behavior_engine.update_config(
            config.behaviorSettings,
            target_map
        )
        print("钉钉机器人: 行为配置已热更新，计时器已重置")

class DingtalkInternalHandler(dingtalk_stream.ChatbotHandler):
    def __init__(self, bot_logic):
        super(DingtalkInternalHandler, self).__init__()
        self.bot_logic = bot_logic

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        try:
            # 解析原始消息
            incoming_message = ChatbotMessage.from_dict(callback.data)
            # 传入完整数据 callback.data 以便解析更多隐藏字段
            await self.bot_logic.on_message(callback.data, incoming_message, self)
        except Exception as e:
            print(f"消息处理异常: {e}")
        return AckMessage.STATUS_OK, 'OK'

class DingtalkClientLogic:
    def __init__(self, config):
        self.config = config
        self.memoryList = {}
        self.port = get_port()
        self.separators = config.separators if config.separators else ['。', '\n', '？', '！']
        
        # --- 新增：注册到行为引擎 ---
        # 告知引擎：dingtalk 平台的执行逻辑由我负责
        global_behavior_engine.register_handler("dingtalk", self.execute_behavior_event)

    async def _get_image_base64(self, url: str) -> Optional[str]:
        """下载钉钉图片并转换为 Base64，解决 AI 访问 403 问题"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.read()
                        return base64.b64encode(data).decode('utf-8')
                    else:
                        print(f"图片下载失败, HTTP状态码: {response.status}")
        except Exception as e:
            print(f"图片处理异常: {e}")
        return None

    async def on_message(self, raw_data: dict, incoming_message: ChatbotMessage, handler: DingtalkInternalHandler):
        cid = incoming_message.conversation_id
        msg_type = incoming_message.message_type
        global_behavior_engine.report_activity("dingtalk", cid)
        user_text_parts = []  # 收集所有文本片段
        user_content_items = []  # 构造 OpenAI 格式的 content
        has_image = False
        
        # --- A. 增强型消息解析 ---
        
        # 1. 处理纯文本消息
        if msg_type == "text":
            if hasattr(incoming_message, 'text') and incoming_message.text:
                user_text_parts.append(incoming_message.text.content.strip())
        
        # 2. 处理纯图片消息
        elif msg_type == "picture":
            download_code = incoming_message.image_content.download_code
            if download_code:
                img_url = handler.get_image_download_url(download_code)
                if img_url:
                    base64_str = await self._get_image_base64(img_url)
                    if base64_str:
                        has_image = True
                        user_content_items.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                        })
            # 尝试获取图片附带文字
            if hasattr(incoming_message, 'text') and incoming_message.text:
                user_text_parts.append(incoming_message.text.content.strip())
            elif raw_data.get("content", {}).get("text"):
                user_text_parts.append(raw_data["content"]["text"].strip())
        
        # 3. 处理富文本消息（同时包含文字和图片）
        elif msg_type == "richText":
            # 关键修正：SDK 中定义的是 rich_text_content
            if hasattr(incoming_message, 'rich_text_content') and incoming_message.rich_text_content:
                rich_list = incoming_message.rich_text_content.rich_text_list
                
                if rich_list:
                    for item in rich_list:
                        # 提取文本
                        if 'text' in item and item['text']:
                            user_text_parts.append(item['text'])
                        
                        # 提取图片
                        if 'downloadCode' in item and item['downloadCode']:
                            download_code = item['downloadCode']
                            img_url = handler.get_image_download_url(download_code)
                            if img_url:
                                base64_str = await self._get_image_base64(img_url)
                                if base64_str:
                                    has_image = True
                                    user_content_items.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                                    })
            
            # 备用：如果 SDK 解析失败，尝试从 raw_data 解析
            if not user_text_parts and not user_content_items:
                content = raw_data.get('content', {})
                if 'richText' in content:
                    for item in content['richText']:
                        if 'text' in item:
                            user_text_parts.append(item['text'])
                        if 'downloadCode' in item:
                            # 同样的图片处理逻辑...
                            download_code = item['downloadCode']
                            img_url = handler.get_image_download_url(download_code)
                            if img_url:
                                base64_str = await self._get_image_base64(img_url)
                                if base64_str:
                                    has_image = True
                                    user_content_items.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                                    })

        # 合并所有文本
        user_text = "\n".join(user_text_parts).strip()
        
        # --- B. 指令与过滤（保持原逻辑）---
        if not user_text and not has_image:
            return

        # 在 on_message 方法中
        if "/id" in user_text.lower():
            # 判断当前是群聊还是单聊
            if cid.startswith("cid"):
                # --- 情况 A: 群聊 ---
                msg = (
                    f"【当前为群聊】\n"
                    f"群会话 ID (OpenConversationId):\n`{cid}`\n\n"
                    f"请复制上方 ID 填入目标列表，机器人将向**本群**推送消息。"
                )
            else:
                # --- 情况 B: 单聊 ---
                # 优先获取企业内部 StaffId
                staff_id = getattr(incoming_message, 'sender_staff_id', None)
                if not staff_id:
                    staff_id = raw_data.get("senderStaffId")
                
                # 兜底：如果你之前测出来的 0246... 那个ID能用，也可以显示 sender_id
                final_id = staff_id if staff_id else incoming_message.sender_id

                msg = (
                    f"【当前为单聊】\n"
                    f"您的用户 ID (UserID):\n`{final_id}`\n\n"
                    f"请复制上方 ID 填入目标列表，机器人将向**您个人**推送消息。"
                )
            
            handler.reply_markdown("ID 抓取助手", msg, incoming_message)
            return

        if self.config.quickRestart and user_text and ("/重启" in user_text or "/restart" in user_text):
            self.memoryList[cid] = []
            handler.reply_text("对话记录已重置。", incoming_message)
            return
        
        if self.config.wakeWord and self.config.wakeWord not in user_text and not has_image:
            return

        # --- C. 构造 OpenAI 消息格式 ---
        if cid not in self.memoryList: 
            self.memoryList[cid] = []
        
        current_content = []
        if user_text:
            current_content.append({"type": "text", "text": user_text})
        
        # 插入图片（OpenAI 格式要求图片在文本之前或混合插入，这里放在文本后也可以）
        if has_image:
            current_content.extend(user_content_items)
            if not user_text:
                current_content.insert(0, {"type": "text", "text": "请分析这张图片"})

        self.memoryList[cid].append({"role": "user", "content": current_content})

        # --- D. AI 调用与流式输出 ---
        ai_client = AsyncOpenAI(api_key="none", base_url=f"http://127.0.0.1:{self.port}/v1")
        state = {"text_buffer": "", "full_response": ""}
        
        try:
            stream = await ai_client.chat.completions.create(
                model=self.config.DingtalkAgent,
                messages=self.memoryList[cid],
                stream=True
            )

            async for chunk in stream:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                
                # 处理推理内容 (如 DeepSeek R1)
                reasoning = ""
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if self.config.reasoningVisible:
                        reasoning = delta.reasoning_content
                
                content = delta.content or ""
                combined_chunk = reasoning + content
                
                if not combined_chunk:
                    continue

                state["text_buffer"] += combined_chunk
                state["full_response"] += content

                # 检查分段符，流式回复钉钉
                if any(sep in state["text_buffer"] for sep in self.separators):
                    if state["text_buffer"].strip():
                        handler.reply_markdown("AI 助手", state["text_buffer"], incoming_message)
                    state["text_buffer"] = ""

            # 扫尾
            if state["text_buffer"].strip():
                handler.reply_markdown("AI 助手", state["text_buffer"], incoming_message)

            # --- E. 记忆持久化与裁剪 ---
            self.memoryList[cid].append({"role": "assistant", "content": state["full_response"]})
            if self.config.memoryLimit > 0:
                # 保持 memoryLimit 组对话 (1 user + 1 assistant = 2条)
                while len(self.memoryList[cid]) > self.config.memoryLimit * 2:
                    self.memoryList[cid].pop(0)

        except Exception as e:
            print(f"钉钉 AI 生成异常: {e}")
            handler.reply_text(f"抱歉，处理消息时出错: {str(e)}", incoming_message)

    async def _get_access_token(self) -> Optional[str]:
        """获取钉钉 OpenAPI 的访问令牌"""
        url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
        payload = {
            "appKey": self.config.appKey,
            "appSecret": self.config.appSecret
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("accessToken")
                    else:
                        print(f"钉钉获取Token失败: {await resp.text()}")
        except Exception as e:
            print(f"钉钉获取Token异常: {e}")
        return None
    
    async def execute_behavior_event(self, chat_id: str, behavior_item: BehaviorItem):
        """
        【终极全能版】自动识别 ID 类型并推送
        - 如果是 cid 开头 -> 调群聊接口
        - 如果是其他 -> 调单聊 batchSend 接口
        """
        # 清理 ID (去掉可能存在的空格)
        target_id = str(chat_id).strip()
        if not target_id: return

        logging.info(f"[Dingtalk] 触发主动行为! 目标: {target_id}")

        # --- 1. 准备 AI 回复内容 (保持不变) ---
        def resolve_prompt(behavior):
            action = behavior.action
            if action.type == "prompt": return action.prompt
            elif action.type == "random":
                events = action.random.events
                if not events: return None
                return random.choice(events) if action.random.type == "random" else events[action.random.orderIndex % len(events)]

        prompt_content = resolve_prompt(behavior_item)
        if not prompt_content: return

        try:
            # --- 2. 调用 AI 生成文本 (保持不变) ---
            ai_client = AsyncOpenAI(api_key="none", base_url=f"http://127.0.0.1:{self.port}/v1")
            response = await ai_client.chat.completions.create(
                model=self.config.DingtalkAgent,
                messages=[{"role": "user", "content": "[system]: "+prompt_content}],
                stream=False
            )
            reply_content = response.choices[0].message.content
            if not reply_content: return

            # --- 3. 获取 Token ---
            token = await self._get_access_token()
            if not token: return
            
            headers = {
                "x-acs-dingtalk-access-token": token,
                "Content-Type": "application/json"
            }

            # --- 4. 核心：智能分流逻辑 ---
            if target_id.startswith("cid"):
                # ============ 分支 A: 群聊推送 ============
                logging.info(f"[Dingtalk] 识别为群聊 ID，调用 groupMessages/send")
                url = "https://api.dingtalk.com/v1.0/robot/groupMessages/send"
                payload = {
                    "msgKey": "sampleMarkdown",
                    "msgParam": json.dumps({
                        "title": "AI 助手",
                        "text": reply_content
                    }),
                    "openConversationId": target_id, # 群 ID
                    "robotCode": self.config.appKey
                }
            else:
                # ============ 分支 B: 单聊推送 (BatchSend) ============
                logging.info(f"[Dingtalk] 识别为用户 ID，调用 oToMessages/batchSend")
                url = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"
                
                # 注意：batchSend 的 msgParam 必须是字符串格式的 JSON
                param_str = json.dumps({
                    "title": "AI 助手",
                    "text": reply_content
                })
                
                payload = {
                    "robotCode": self.config.appKey,
                    "userIds": [target_id],    # 用户 ID 列表
                    "msgKey": "sampleMarkdown",
                    "msgParam": param_str      # 这是一个字符串
                }

            # --- 5. 发送请求 ---
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    result = await resp.json()
                    
                    # 成功判断：
                    # 群聊返回 processQueryKey，单聊也返回 processQueryKey
                    if resp.status == 200 and result.get("processQueryKey"):
                        logging.info(f"[Dingtalk] 推送成功! 目标: {target_id}")
                        # 写入记忆
                        if target_id not in self.memoryList: self.memoryList[target_id] = []
                        self.memoryList[target_id].append({"role": "assistant", "content": reply_content})
                    else:
                        logging.error(f"[Dingtalk] 推送失败: {result}")
                        
        except Exception as e:
            logging.error(f"[Dingtalk] 执行异常: {e}")       