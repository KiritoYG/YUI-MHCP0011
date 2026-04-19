import asyncio
import base64
import io
import json
import logging
import random
import re
import threading
import weakref
from typing import Dict, List, Optional, Any

import aiohttp
import discord
from discord.ext import commands, tasks
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from py.behavior_engine import BehaviorItem, BehaviorSettings,global_behavior_engine
from py.get_setting import convert_to_opus_simple, get_port, load_settings

# ------------------ 配置模型 ------------------
class DiscordBotConfig(BaseModel):
    token: str
    llm_model: str = "super-model"
    memory_limit: int = 10
    separators: List[str] = ["。", "\n", "？", "！"]
    reasoning_visible: bool = False
    quick_restart: bool = True
    enable_tts: bool = True
    wakeWord: str              # 唤醒词
    # --- 新增：行为规则设置 ---
    behaviorSettings: Optional[BehaviorSettings] = None
    # Discord 特定的推送目标 ID 列表 (Channel IDs)
    behaviorTargetChatIds: List[str] = Field(default_factory=list)

# ------------------ 管理器 ------------------
class DiscordBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_client: Optional["DiscordClient"] = None
        self.is_running = False
        self.config: Optional[DiscordBotConfig] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = threading.Event()
        self._ready_complete = threading.Event()
        self._startup_error: Optional[str] = None
        self._stop_requested = False

    # ---------- 生命周期 ----------
    def start_bot(self, config: DiscordBotConfig):
        if self.is_running:
            raise RuntimeError("Discord 机器人已在运行")
        self.config = config
        self._shutdown_event.clear()
        self._ready_complete.clear()
        self._startup_error = None
        self._stop_requested = False

        self.bot_thread = threading.Thread(
            target=self._run_bot_thread, args=(config,), daemon=True, name="DiscordBotThread"
        )
        self.bot_thread.start()

        if not self._ready_complete.wait(timeout=30):
            self.stop_bot()
            raise RuntimeError("Discord 机器人就绪超时")

        if self._startup_error:
            self.stop_bot()
            raise RuntimeError(f"Discord 机器人启动失败: {self._startup_error}")

    def _run_bot_thread(self, config: DiscordBotConfig):
        """线程中运行 Discord 机器人"""
        try:
            # 1. 创建并设置循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # 2. 定义一个统一的异步启动函数
            async def main_startup():
                try:
                    # 在异步环境下加载设置，避免 asyncio.run 冲突
                    settings = await load_settings()
                    behavior_data = settings.get("behaviorSettings", {})
                    
                    target_ids = config.behaviorTargetChatIds
                    if not target_ids:
                        discord_conf = settings.get("discordBotConfig", {})
                        target_ids = discord_conf.get("behaviorTargetChatIds", [])
                    
                    if behavior_data:
                        logging.info(f"Discord 线程: 同步行为配置... 目标频道数: {len(target_ids)}")
                        target_map = {"discord": target_ids}
                        global_behavior_engine.update_config(behavior_data, target_map)
                        
                        # 更新本地配置对象
                        config.behaviorSettings = behavior_data if isinstance(behavior_data, BehaviorSettings) else BehaviorSettings(**behavior_data)
                        config.behaviorTargetChatIds = target_ids

                    # 3. 实例化 Client
                    self.bot_client = DiscordClient(config, manager=self)

                    # 4. 启动行为引擎 (此时在运行的 loop 中，可以使用 create_task)
                    if not global_behavior_engine.is_running:
                        asyncio.create_task(global_behavior_engine.start())
                        logging.info("行为引擎已在 Discord 线程启动")

                    # 5. 启动 Discord Bot (这会阻塞直到 Bot 关闭)
                    await self.bot_client.start(config.token)
                except Exception as e:
                    self._startup_error = str(e)
                    logging.exception("Discord 机器人启动过程中出错")

            # 运行异步主任务
            self.loop.run_until_complete(main_startup())

        except Exception as e:
            if not self._stop_requested:
                self._startup_error = str(e)
                logging.exception("Discord 机器人线程异常")
        finally:
            self._cleanup()

    def stop_bot(self):
        if not self.is_running and not self.bot_thread:
            return
        self._stop_requested = True
        self._shutdown_event.set()
        self.is_running = False
        if self.bot_client:
            asyncio.run_coroutine_threadsafe(self.bot_client.close(), self.loop)
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=5)
        self._cleanup()

    def _cleanup(self):
        self.is_running = False
        if self.loop and not self.loop.is_closed():
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.close()
            except Exception:
                pass
        logging.info("Discord 机器人资源已清理")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "thread_alive": self.bot_thread.is_alive() if self.bot_thread else False,
            "ready_completed": self._ready_complete.is_set(),
            "startup_error": self._startup_error,
            "config": self.config.model_dump() if self.config else None,
        }

    def update_behavior_config(self, config: DiscordBotConfig):
        """
        热更新行为配置，不重启机器人
        """
        # 更新 Manager 的本地记录
        self.config = config
        
        # 1. 更新 Client 内部的实时参数
        if self.bot_client:
            self.bot_client.config.llm_model = config.llm_model 
            self.bot_client.config.enable_tts = config.enable_tts
            self.bot_client.config.wakeWord = config.wakeWord

        # 2. 更新全局行为引擎
        target_map = {
            "discord": config.behaviorTargetChatIds
        }
        
        global_behavior_engine.update_config(
            config.behaviorSettings,
            target_map
        )
        logging.info("Discord 机器人: 行为配置已热更新，计时器已重置")

# ------------------ Discord Client ------------------
class DiscordClient(discord.Client):
    def __init__(self, config: DiscordBotConfig, manager: DiscordBotManager):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.config = config
        self.manager = manager
        self.memory: Dict[int, List[dict]] = {}  # channel_id -> msgs
        self.async_tools: Dict[int, List[str]] = {}
        self.file_links: Dict[int, List[str]] = {}
        self._shutdown_requested = False
        
        # --- 新增：注册到行为引擎 ---
        # 告知引擎：Discord 平台的执行逻辑由本实例负责
        global_behavior_engine.register_handler("discord", self.execute_behavior_event)

    async def on_ready(self):
        self.manager.is_running = True
        self.manager._ready_complete.set()
        logging.info(f"✅ Discord 机器人已上线：{self.user}")

    async def on_message(self, msg: discord.Message):
        if self._shutdown_requested or msg.author == self.user:
            return
        # 统一入口
        try:
            await self._handle_message(msg)
        except Exception as e:
            logging.exception("处理 Discord 消息失败")
            await msg.channel.send(f"处理消息失败：{e}")

    # ---------- 消息主处理 ----------
    async def _handle_message(self, msg: discord.Message):
        cid = msg.channel.id
        if cid not in self.memory:
            self.memory[cid] = []
            self.async_tools[cid] = []
            self.file_links[cid] = []

        # --- 新增：上报活跃状态到引擎，用于无输入检测 ---
        global_behavior_engine.report_activity("discord", str(cid))

        # 1. 指令处理
        if msg.content:
            content_strip = msg.content.strip()
            
            # [新增] /id 指令：获取当前频道 ID
            if content_strip.lower() == "/id":
                info_msg = (
                    f"🤖 **Discord Session Information Identified Successfully**\n\n"
                    f"Current Channel ID:\n`{cid}`\n\n"
                    f"💡 Note: Please directly copy the ID above and fill it into the Discord target list in the 'Autonomous Actions' section of the backend."
                )
                await msg.reply(info_msg)
                return

            # 快速重启
            if self.config.quick_restart:
                if content_strip in {"/重启", "/restart"}:
                    self.memory[cid].clear()
                    await msg.reply("对话记录已重置。")
                    return

        # 2. 拼装用户内容
        user_content = []
        user_text = ""
        has_media = False

        # 2.1 文本
        if msg.content:
            user_text = msg.content

        # 2.2 图片
        for att in msg.attachments:
            if att.content_type and att.content_type.startswith("image"):
                b64data = base64.b64encode(await att.read()).decode()
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{att.content_type};base64,{b64data}"}
                })
                has_media = True

        # 2.3 语音
        for att in msg.attachments:
            if att.content_type and att.content_type.startswith("audio"):
                audio_bytes = await att.read()
                asr_text = await self._transcribe_audio(audio_bytes, att.filename)
                if asr_text:
                    user_text += f"\n[语音转写] {asr_text}"
                else:
                    user_text += "\n[语音转写失败]"
        
        if self.config.wakeWord:
            if self.config.wakeWord not in user_text:
                logging.info(f"未检测到唤醒词: {self.config.wakeWord}")
                return

        if has_media and user_text:
            user_content.append({"type": "text", "text": user_text})
        if not has_media and not user_text:
            return

        self.memory[cid].append({"role": "user", "content": user_content or user_text})

        # 3. 请求 LLM (后续逻辑保持不变...)
        settings = await load_settings()
        client = AsyncOpenAI(api_key="super-secret-key", base_url=f"http://127.0.0.1:{get_port()}/v1")

        async_tools = self.async_tools.get(cid, [])
        file_links = self.file_links.get(cid, [])

        try:
            stream = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=self.memory[cid],
                stream=True,
                extra_body={
                    "asyncToolsID": async_tools,
                    "fileLinks": file_links,
                    "is_app_bot": True,
                },
            )
        except Exception as e:
            logging.warning(f"LLM 请求失败: {e}")
            await msg.channel.send("LLM 响应超时，请稍后再试。")
            return

        # 4. 流式解析 (省略已有代码)
        state = {
            "text_buffer": "", 
            "image_buffer": "", 
            "image_cache": [],
            "audio_buffer": [] 
        }
        full_response = []

        async for chunk in stream:
            if not chunk.choices: continue
            delta_raw = chunk.choices[0].delta
            if hasattr(delta_raw, "audio") and delta_raw.audio:
                if "data" in delta_raw.audio:
                    state["audio_buffer"].append(delta_raw.audio["data"])
            reasoning_content = getattr(delta_raw, "reasoning_content", None) or ""
            async_tool_id = getattr(delta_raw, "async_tool_id", None) or ""
            tool_link = getattr(delta_raw, "tool_link", None) or ""
            if tool_link and settings.get("tools", {}).get("toolMemorandum", {}).get("enabled"):
                if tool_link not in self.file_links[cid]: self.file_links[cid].append(tool_link)
            if async_tool_id:
                if async_tool_id not in self.async_tools[cid]: self.async_tools[cid].append(async_tool_id)
                else: self.async_tools[cid].remove(async_tool_id)
            content = delta_raw.content or ""
            if reasoning_content and self.config.reasoning_visible: content = reasoning_content
            full_response.append(content)
            state["text_buffer"] += content
            state["image_buffer"] += content
            if state["text_buffer"]:
                force_split = len(state["text_buffer"]) > 1800
                while True:
                    buffer = state["text_buffer"]
                    split_pos = -1
                    in_code_block = False
                    if force_split:
                        min_idx = len(buffer) + 1
                        found_sep_len = 0
                        for sep in self.config.separators:
                            idx = buffer.find(sep)
                            if idx != -1 and idx < min_idx:
                                min_idx = idx
                                found_sep_len = len(sep)
                        if min_idx <= len(buffer): split_pos = min_idx + found_sep_len
                    else:
                        i = 0
                        while i < len(buffer):
                            if buffer[i:].startswith("```"): in_code_block = not in_code_block; i += 3; continue
                            if not in_code_block:
                                found_sep = False
                                for sep in self.config.separators:
                                    if buffer[i:].startswith(sep): split_pos = i + len(sep); found_sep = True; break
                                if found_sep: break
                            i += 1
                    if split_pos == -1: break
                    seg = buffer[:split_pos]
                    state["text_buffer"] = buffer[split_pos:]
                    seg = self._clean_text(seg)
                    if seg and not self.config.enable_tts: await self._send_segment(msg, seg)
                    if force_split: break
        if state["text_buffer"]:
            seg = self._clean_text(state["text_buffer"])
            if seg and not self.config.enable_tts: await self._send_segment(msg, seg)
        self._extract_images(state)
        for img_url in state["image_cache"]: await self._send_image(msg, img_url)
        has_omni_audio = False
        if state["audio_buffer"]:
            try:
                full_audio_b64 = "".join(state["audio_buffer"])
                raw_audio_bytes = base64.b64decode(full_audio_b64)
                final_audio, is_opus = await asyncio.to_thread(convert_to_opus_simple, raw_audio_bytes)
                await self._send_omni_voice(msg, final_audio, is_opus)
                has_omni_audio = True
            except Exception as e: logging.error(f"Omni 音频处理失败: {e}")
        full_content = "".join(full_response)
        if self.config.enable_tts and not has_omni_audio: await self._send_voice(msg, full_content)
        self.memory[cid].append({"role": "assistant", "content": full_content})
        if self.config.memory_limit > 0:
            while len(self.memory[cid]) > self.config.memory_limit * 2: self.memory[cid].pop(0)
    
    # [新增] 发送 Omni 语音
    async def _send_omni_voice(self, msg: discord.Message, audio_data: bytes, is_opus: bool):
        """发送 Omni 模型生成的音频文件"""
        try:
            # Discord 没有专门的 Voice Message API，通常作为文件附件发送
            ext = "opus" if is_opus else "wav"
            filename = f"voice.{ext}"
            
            # 创建 Discord 文件对象
            file = discord.File(io.BytesIO(audio_data), filename=filename)
            
            # 回复消息
            await msg.reply(file=file, mention_author=False)
            logging.info(f"已发送 Omni 音频: {filename}")
        except Exception as e:
            logging.error(f"发送 Omni 音频异常: {e}")

    # ---------- 工具 ----------
    async def _transcribe_audio(self, audio_bytes: bytes, filename: str) -> Optional[str]:
        form = aiohttp.FormData()
        form.add_field("audio", io.BytesIO(audio_bytes), filename=filename, content_type="audio/ogg")
        form.add_field("format", "auto")
        async with aiohttp.ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{get_port()}/asr", data=form) as r:
                if r.status != 200:
                    return None
                res = await r.json()
                return res.get("text") if res.get("success") else None

    def _clean_text(self, text: str) -> str:
        # 1. 移除 Markdown 图片 ![alt](url) -> 空
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # 移除html标签
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

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

    async def _send_segment(self, msg: discord.Message, seg: str):
        if self.config.enable_tts:
            pass
        else:
            await msg.channel.send(seg)

    async def _send_voice(self, msg: discord.Message, text: str):
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
            "mobile_optimized": True,  
            "format": "opus"           # 明确请求opus格式
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{get_port()}/tts", json=payload) as r:
                if r.status != 200:
                    await msg.channel.send("语音生成失败")
                    return
                opus = await r.read()
                file = discord.File(io.BytesIO(opus), filename="voice.opus")
                await msg.channel.send(file=file)

    async def close(self):
        self._shutdown_requested = True
        await super().close()

    def _extract_images(self, state: Dict[str, Any]):
        """从缓冲区提取 markdown 图片链接"""
        buffer = state["image_buffer"]
        pattern = r'!\[.*?\]\((https?://[^\s)]+)'
        for m in re.finditer(pattern, buffer):
            state["image_cache"].append(m.group(1))

    async def _send_image(self, msg: discord.Message, img_url: str):
        """下载并发送图片到当前频道"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(img_url) as resp:
                    if resp.status != 200:
                        logging.warning(f"下载图片失败: {img_url}")
                        return
                    data = await resp.read()
                    ext = img_url.split("?")[0].split(".")[-1][:4]  # 简单取后缀
                    ext = ext if ext.lower() in {"png", "jpg", "jpeg", "gif", "webp"} else "png"
                    file = discord.File(io.BytesIO(data), filename=f"image.{ext}")
                    await msg.channel.send(file=file)
        except Exception as e:
            logging.exception(f"发送图片失败: {img_url}")

    async def execute_behavior_event(self, chat_id: str, behavior_item: BehaviorItem):
        """
        回调函数：响应行为引擎的指令
        """
        logging.info(f"[DiscordClient] 行为触发! 目标: {chat_id}, 动作类型: {behavior_item.action.type}")
        
        prompt_content = await self._resolve_behavior_prompt(behavior_item)
        if not prompt_content: return

        cid = int(chat_id)
        if cid not in self.memory:
            self.memory[cid] = []
        
        # 构造上下文：历史记录 + 系统指令
        messages = self.memory[cid].copy()
        system_instruction = f"[system]: {prompt_content}"
        messages.append({"role": "user", "content": system_instruction})
        
        # 同步到内存，维持逻辑连贯
        self.memory[cid].append({"role": "user", "content": system_instruction})

        try:
            client = AsyncOpenAI(
                api_key="super-secret-key",
                base_url=f"http://127.0.0.1:{get_port()}/v1"
            )
            
            # 使用非流式请求处理主动行为，便于逻辑简化
            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                stream=False, 
                extra_body={
                    "is_app_bot": True,
                    "behavior_trigger": True
                }
            )
            
            reply_content = response.choices[0].message.content
            if reply_content:
                channel = self.get_channel(cid)
                if channel:
                    # 1. 发送文本
                    await channel.send(reply_content)
                    self.memory[cid].append({"role": "assistant", "content": reply_content})
                    
                    # 2. 如果开启了 TTS，则发送语音
                    if self.config.enable_tts:
                        # 构造 MockMessage 以复用现有 TTS 函数
                        class MockMsg:
                            def __init__(self, c): self.channel = c
                        await self._send_voice(MockMsg(channel), reply_content)
            
        except Exception as e:
            logging.error(f"[DiscordClient] 执行行为 API 调用失败: {e}")   

    async def _resolve_behavior_prompt(self, behavior: BehaviorItem) -> str:
        """解析行为配置，生成具体的 Prompt 指令"""
        from py.random_topic import get_random_topics
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