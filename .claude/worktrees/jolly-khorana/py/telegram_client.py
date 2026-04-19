import asyncio, aiohttp, io, base64, json, logging, re, time
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from py.behavior_engine import BehaviorItem
from py.get_setting import convert_to_opus_simple, get_port, load_settings

class TelegramClient:
    def __init__(self):
        self.TelegramAgent = "super-model"
        self.memoryLimit = 10
        self.memoryList: Dict[int, List[Dict]] = {}  # chat_id -> messages
        self.asyncToolsID: Dict[int, List[str]] = {}
        self.fileLinks: Dict[int, List[str]] = {}
        self.separators = ["。", "\n", "？", "！"]
        self.reasoningVisible = False
        self.quickRestart = True
        self.enableTTS = False
        self.wakeWord = None
        self.bot_token: str = ""
        self.config = None # 存储 config 引用
        self._is_ready = False
        self._manager_ref = None
        self._ready_callback = None
        self._shutdown_requested = False
        self.offset = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self.port = get_port()
        
        # --- 新增：注册到行为引擎 ---
        from py.behavior_engine import global_behavior_engine
        global_behavior_engine.register_handler("telegram", self.execute_behavior_event)

    # -------------------- 生命周期 --------------------
    async def run(self):
        # Add a session timeout slightly above the polling timeout
        timeout = aiohttp.ClientTimeout(total=35)  # 5s buffer
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        self._is_ready = True
        if self._manager_ref:
            manager = self._manager_ref()
            if manager:
                manager._ready_complete.set()
                manager.is_running = True

        logging.info("Telegram 轮询开始")
        try:
            while not self._shutdown_requested:
                try:
                    updates = await self._get_updates()
                    for u in updates:
                        await self._handle_update(u)
                except asyncio.TimeoutError:
                    # Normal when shutdown happens during long poll
                    pass
                
                # Prevent tight loop when no updates
                if not updates:
                    await asyncio.sleep(0.1)
        finally:
            await self.session.close()

    async def _get_updates(self):
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        # CRITICAL: Reduce from 30s to 5s for responsive shutdown
        async with self.session.get(url, params={"offset": self.offset, "timeout": 5}) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            if not data.get("ok"):
                return []
            return data["result"]

    # -------------------- 消息入口 --------------------
    async def _handle_update(self, u: dict):
        if "message" not in u:
            return
        msg = u["message"]
        self.offset = u["update_id"] + 1
        chat_id = msg["chat"]["id"]

        # 文字
        if "text" in msg:
            await self._handle_text(chat_id, msg)
        # 图片（任意尺寸）
        elif "photo" in msg:
            await self._handle_photo(chat_id, msg)
        # 语音 / 音频
        elif "voice" in msg or "audio" in msg:
            await self._handle_voice(chat_id, msg)

    # -------------------- 文字 --------------------
    async def _handle_text(self, chat_id: int, msg: dict):
        text = msg["text"]
        
        # --- 新增：上报活跃状态到引擎，用于无输入检测 ---
        from py.behavior_engine import global_behavior_engine
        global_behavior_engine.report_activity("telegram", str(chat_id))

        if self.quickRestart:
            if text in {"/restart", "/重启"}:
                self.memoryList[chat_id] = []
                await self._send_text(chat_id, "对话记录已重置。")
                return

        # --- 新增：/id 指令 ---
        if text.strip().lower() == "/id":
            info_msg = (
                f"🤖 **Telegram Session Information Identified Successfully**\n\n"
                f"Current Chat ID:\n`{chat_id}`\n\n"
                f"💡 Note: Please directly copy the ID above and paste it into the 'Autonomous Actions' Telegram target list in the backend."
            )
            await self._send_text(chat_id, info_msg)
            return

        if self.wakeWord:
            if self.wakeWord not in text:
                logging.info(f"未检测到唤醒词: {self.wakeWord}")
                return
        await self._process_llm(chat_id, text, [], msg.get("message_id"))

    # -------------------- 图片 --------------------
    async def _handle_photo(self, chat_id: int, msg: dict):
        from py.behavior_engine import global_behavior_engine
        global_behavior_engine.report_activity("telegram", str(chat_id))
        photos = msg["photo"]  # 数组，尺寸升序
        file_id = photos[-1]["file_id"]
        file_info = await self._get_file(file_id)
        if not file_info:
            await self._send_text(chat_id, "下载图片失败")
            return
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_info['file_path']}"
        async with self.session.get(url) as resp:
            if resp.status != 200:
                await self._send_text(chat_id, "下载图片失败")
                return
            img_bytes = await resp.read()
        base64_data = base64.b64encode(img_bytes).decode()
        data_uri = f"data:image/jpeg;base64,{base64_data}"
        user_content = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": "用户发送了一张图片"}
        ]
        await self._process_llm(chat_id, "", user_content, msg.get("message_id"))

    # -------------------- 语音 --------------------
    async def _handle_voice(self, chat_id: int, msg: dict):
        from py.behavior_engine import global_behavior_engine
        global_behavior_engine.report_activity("telegram", str(chat_id))
        voice = msg.get("voice") or msg.get("audio")
        file_id = voice["file_id"]
        file_info = await self._get_file(file_id)
        if not file_info:
            await self._send_text(chat_id, "下载语音失败")
            return
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_info['file_path']}"
        async with self.session.get(url) as resp:
            if resp.status != 200:
                await self._send_text(chat_id, "下载语音失败")
                return
            voice_bytes = await resp.read()
        # 调用本地 ASR
        text = await self._transcribe(voice_bytes)
        if self.wakeWord:
            if self.wakeWord not in text:
                logging.info(f"未检测到唤醒词: {self.wakeWord}")
                return

        if not text:
            await self._send_text(chat_id, "语音转文字失败")
            return  
        await self._process_llm(chat_id, text, [], msg.get("message_id"))

    # -------------------- LLM 统一处理 --------------------
    async def _process_llm(self, chat_id: int, text: str, extra_content: List[dict], reply_to_msg_id: Optional[int]):
        if chat_id not in self.memoryList:
            self.memoryList[chat_id] = []
        if chat_id not in self.asyncToolsID:
            self.asyncToolsID[chat_id] = []
        if chat_id not in self.fileLinks:
            self.fileLinks[chat_id] = []

        # 构造 user 消息
        if extra_content:
            user_msg = {"role": "user", "content": extra_content}
        else:
            user_msg = {"role": "user", "content": text}
        self.memoryList[chat_id].append(user_msg)

        settings = await load_settings()
        client = AsyncOpenAI(api_key="super-secret-key", base_url=f"http://127.0.0.1:{get_port()}/v1")

        # 初始化状态，新增 audio_buffer
        state = {
            "text_buffer": "", 
            "image_cache": [],
            "audio_buffer": [] # <--- 新增
        }
        full_response = []

        try:
            stream = await client.chat.completions.create(
                model=self.TelegramAgent,
                messages=self.memoryList[chat_id],
                stream=True,
                extra_body={
                    "asyncToolsID": self.asyncToolsID[chat_id],
                    "fileLinks": self.fileLinks[chat_id],
                    "is_app_bot": True,
                    # 后端根据这个标志决定是否返回音频流
                },
            )
            
            async for chunk in stream:
                if not chunk.choices: continue
                
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', '') or ""
                reasoning = getattr(delta, 'reasoning_content', '') or ""
                tool_link = getattr(delta, 'tool_link', '') or ""
                async_tool_id = getattr(delta, 'async_tool_id', '') or ""

                # --- [新增] 捕获音频流 ---
                if hasattr(delta, "audio") and delta.audio:
                    if "data" in delta.audio:
                        state["audio_buffer"].append(delta.audio["data"])
                # -----------------------

                if tool_link and settings["tools"]["toolMemorandum"]["enabled"]:
                    self.fileLinks[chat_id].append(tool_link)
                if async_tool_id:
                    lst = self.asyncToolsID[chat_id]
                    if async_tool_id not in lst:
                        lst.append(async_tool_id)
                    else:
                        lst.remove(async_tool_id)

                seg = reasoning if self.reasoningVisible and reasoning else content
                state["text_buffer"] += seg
                full_response.append(content)

                # 文本分段发送逻辑 (保持不变)
                if state["text_buffer"]:
                    force_split = len(state["text_buffer"]) > 3500
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
                        
                        send_chunk = buffer[:split_pos]
                        state["text_buffer"] = buffer[split_pos:]
                        
                        clean = self._clean_text(send_chunk)
                        if clean and not self.enableTTS:
                            await self._send_text(chat_id, clean)
                                
                        if force_split: break

            # 发送剩余文本
            if state["text_buffer"]:
                clean = self._clean_text(state["text_buffer"])
                if clean and not self.enableTTS:
                    await self._send_text(chat_id, clean)

            # 提取并发送图片
            self._extract_images("".join(full_response), state)
            for img_url in state["image_cache"]:
                await self._send_photo(chat_id, img_url)

            # --- [新增] 处理 Omni 音频 ---
            has_omni_audio = False
            if state["audio_buffer"]:
                try:
                    logging.info(f"处理 Telegram Omni 音频，分片数: {len(state['audio_buffer'])}")
                    full_audio_b64 = "".join(state["audio_buffer"])
                    raw_audio_bytes = base64.b64decode(full_audio_b64)
                    
                    # 异步转码
                    final_audio, is_opus = await asyncio.to_thread(
                        convert_to_opus_simple, 
                        raw_audio_bytes
                    )
                    
                    # 发送
                    await self._send_omni_voice(chat_id, final_audio, is_opus)
                    has_omni_audio = True
                except Exception as e:
                    logging.error(f"Omni 音频处理失败: {e}")
            # ---------------------------

            # 记忆
            assistant_text = "".join(full_response)
            self.memoryList[chat_id].append({"role": "assistant", "content": assistant_text})

            # 记忆限制
            if self.memoryLimit > 0:
                while len(self.memoryList[chat_id]) > self.memoryLimit * 2:
                    self.memoryList[chat_id].pop(0)
                    if self.memoryList[chat_id]:
                        self.memoryList[chat_id].pop(0)

            # 传统 TTS (如果没有 Omni 音频且开启了 TTS)
            if self.enableTTS and assistant_text and not has_omni_audio:
                await self._send_voice(chat_id, assistant_text)
                
        except Exception as e:
            logging.error(f"LLM 处理异常: {e}")
            await self._send_text(chat_id, f"处理出错: {e}")

    async def _send_omni_voice(self, chat_id: int, audio_data: bytes, is_opus: bool):
        """发送 Omni 语音消息"""
        try:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            
            # 如果是 Opus 格式，可以使用 sendVoice 发送语音气泡
            if is_opus:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendVoice"
                # Telegram 对 filename 没那么严格，但 mime-type 最好正确
                data.add_field("voice", io.BytesIO(audio_data), filename="voice.ogg", content_type="audio/ogg")
                logging.info("发送 Omni 语音气泡 (sendVoice)")
            else:
                # 如果转换失败（例如 Raw PCM 或 WAV），sendVoice 可能会失败或不显示波形
                # 降级使用 sendDocument 发送文件
                url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
                data.add_field("document", io.BytesIO(audio_data), filename="reply.wav")
                logging.info("发送 Omni 音频文件 (sendDocument)")

            async with self.session.post(url, data=data) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logging.error(f"发送 Omni 音频失败: {resp.status} - {err_text}")
        except Exception as e:
            logging.error(f"发送 Omni 音频异常: {e}")


    # -------------------- 发送 API 封装 --------------------
    async def _send_text(self, chat_id: int, text: str, reply_to_msg_id: Optional[int] = None):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        # 1. 尝试使用 Markdown (Legacy) 模式
        # 相比 MarkdownV2，Legacy 模式容错率更高，虽然不支持下划线和删除线，但支持粗体、斜体、代码块和链接
        payload = {
            "chat_id": chat_id, 
            "text": text, 
            "parse_mode": "Markdown" 
        }
        if reply_to_msg_id:
            payload["reply_to_message_id"] = reply_to_msg_id
            
        async with self.session.post(url, json=payload) as resp:
            if resp.status == 200:
                return # 发送成功
            
            # 2. 如果发送失败（通常是因为 Markdown 语法未闭合导致的 400 错误）
            # 读取错误信息（可选，用于调试）
            # err_text = await resp.text() 
            # logging.warning(f"Markdown 发送失败，尝试纯文本重发: {err_text}")

            # 3. 回退策略：移除 parse_mode，发送纯文本
            payload.pop("parse_mode")
            await self.session.post(url, json=payload)

    async def _send_photo(self, chat_id: int, image_url: str):
        # 先下载
        async with self.session.get(image_url) as resp:
            if resp.status != 200:
                return
            img_bytes = await resp.read()
        #  multipart 上传
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        data.add_field("photo", io.BytesIO(img_bytes), filename="image.jpg")
        await self.session.post(url, data=data)

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

    async def _send_voice(self, chat_id: int, text: str):
        from py.get_setting import load_settings
        settings = await load_settings()
        tts_settings = settings.get("ttsSettings", {})
        index = 0
        text = self.clean_markdown(text)
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
                    logging.error(f"TTS 请求失败: {resp.status}")
                    error_text = await resp.text()
                    logging.error(f"TTS 错误响应: {error_text}")
                    await self._send_text(chat_id, "语音生成失败，请稍后重试")
                    return

                opus_data = await resp.read()
                audio_format = resp.headers.get("X-Audio-Format", "unknown")
                
                logging.info(f"TTS响应成功，opus大小: {len(opus_data) / 1024:.1f}KB，格式: {audio_format}")
        # 上传语音
        url = f"https://api.telegram.org/bot{self.bot_token}/sendVoice"
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        data.add_field("voice", io.BytesIO(opus_data), filename="voice.opus")
        await self.session.post(url, data=data)

    # -------------------- 工具 --------------------
    async def _get_file(self, file_id: str) -> Optional[dict]:
        url = f"https://api.telegram.org/bot{self.bot_token}/getFile"
        async with self.session.get(url, params={"file_id": file_id}) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("result")

    async def _transcribe(self, audio_bytes: bytes) -> Optional[str]:
        form = aiohttp.FormData()
        form.add_field("audio", io.BytesIO(audio_bytes), filename="voice.ogg")
        form.add_field("format", "auto")
        async with self.session.post(f"http://127.0.0.1:{get_port()}/asr", data=form) as resp:
            if resp.status != 200:
                return None
            res = await resp.json()
            return res.get("text") if res.get("success") else None

    def _clean_text(self, text: str) -> str:
        # 1. 移除 Markdown 图片 ![alt](url) -> 空
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # 移除html标签
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def _extract_images(self, full_text: str, state: dict):
        for m in re.finditer(r"!\[.*?\]\((https?://[^\s)]+)", full_text):
            state["image_cache"].append(m.group(1))

    async def execute_behavior_event(self, chat_id: str, behavior_item: BehaviorItem):
        """
        回调函数：响应行为引擎的指令
        """
        logging.info(f"[TelegramClient] 行为触发! 目标: {chat_id}, 动作类型: {behavior_item.action.type}")
        
        prompt_content = await self._resolve_behavior_prompt(behavior_item)
        if not prompt_content: return

        cid = int(chat_id)
        if cid not in self.memoryList:
            self.memoryList[cid] = []
        
        # 构造上下文：历史记录 + 系统指令
        messages = self.memoryList[cid].copy()
        system_instruction = f"[system]: {prompt_content}"
        messages.append({"role": "user", "content": system_instruction})
        
        # 同时记录到内存，维持逻辑连贯
        self.memoryList[cid].append({"role": "user", "content": system_instruction})

        try:
            client = AsyncOpenAI(
                api_key="super-secret-key",
                base_url=f"http://127.0.0.1:{get_port()}/v1"
            )
            
            # 使用非流式请求处理主动行为
            response = await client.chat.completions.create(
                model=self.TelegramAgent,
                messages=messages,
                stream=False, 
                extra_body={
                    "is_app_bot": True,
                    "behavior_trigger": True
                }
            )
            
            reply_content = response.choices[0].message.content
            if reply_content:
                # 1. 发送文本
                await self._send_text(cid, reply_content)
                self.memoryList[cid].append({"role": "assistant", "content": reply_content})
                
                # 2. 如果开启了 TTS，则发送语音
                if self.enableTTS:
                    await self._send_voice(cid, reply_content)
            
        except Exception as e:
            logging.error(f"[TelegramClient] 执行行为 API 调用失败: {e}")

    async def _resolve_behavior_prompt(self, behavior: BehaviorItem) -> Optional[str]:
        """解析行为配置，生成具体的 Prompt 指令"""
        import random
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
                if idx >= len(events): idx = 0
                selected = events[idx]
                action.random.orderIndex = idx + 1 # 内存内更新
                return selected
        return None