import asyncio, threading, weakref, logging, time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from py.behavior_engine import BehaviorSettings
from py.telegram_client import TelegramClient

class TelegramBotConfig(BaseModel):
    TelegramAgent: str        # LLM 模型名
    memoryLimit: int
    separators: list[str]
    reasoningVisible: bool
    quickRestart: bool
    enableTTS: bool
    bot_token: str            # Telegram 必填
    wakeWord: str              # 唤醒词
    # --- 新增：行为规则设置 ---
    behaviorSettings: Optional[BehaviorSettings] = None
    # Telegram 特定的推送目标 ID 列表 (Chat IDs)
    behaviorTargetChatIds: List[str] = Field(default_factory=list)

class TelegramBotManager:
    def __init__(self):
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_client: Optional[TelegramClient] = None
        self.is_running = False
        self.config = None
        self.loop = None
        self._shutdown_event = threading.Event()
        self._startup_complete = threading.Event()
        self._ready_complete = threading.Event()
        self._startup_error: Optional[str] = None
        self._stop_requested = False

    # 以下四个接口与 FeishuBotManager 完全一致，直接复用路由
    def start_bot(self, config: TelegramBotConfig):
        # ADD: Check if previous thread is still alive
        if self.bot_thread and self.bot_thread.is_alive():
            raise Exception("Telegram 机器人线程正在清理中，请稍后再试")
        
        if self.is_running:
            raise Exception("Telegram 机器人已在运行")
        
        self.config = config
        self._shutdown_event.clear()
        self._startup_complete.clear()
        self._ready_complete.clear()
        self._startup_error = None
        self._stop_requested = False

        self.bot_thread = threading.Thread(
            target=self._run_bot_thread, args=(config,), daemon=True, name="TelegramBotThread"
        )
        self.bot_thread.start()

        if not self._startup_complete.wait(timeout=30):
            self.stop_bot()
            raise Exception("Telegram 机器人连接超时")
        if self._startup_error:
            self.stop_bot()
            raise Exception(f"Telegram 机器人启动失败: {self._startup_error}")
        if not self._ready_complete.wait(timeout=30):
            self.stop_bot()
            raise Exception("Telegram 机器人就绪超时")
        if not self.is_running:
            self.stop_bot()
            raise Exception("Telegram 机器人未能正常运行")


    def _run_bot_thread(self, config: TelegramBotConfig):
        # 1. 创建并设置循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # 2. 定义统一的异步启动入口
        async def main_startup():
            try:
                # --- 步骤 A: 异步加载设置 (替代 asyncio.run) ---
                from py.get_setting import load_settings
                from py.behavior_engine import global_behavior_engine, BehaviorSettings
                
                settings = await load_settings()
                behavior_data = settings.get("behaviorSettings", {})
                
                # 获取目标频道列表
                target_ids = config.behaviorTargetChatIds
                if not target_ids:
                    tg_conf = settings.get("telegramBotConfig", {})
                    target_ids = tg_conf.get("behaviorTargetChatIds", [])
                
                # --- 步骤 B: 同步行为配置 ---
                if behavior_data:
                    logging.info(f"Telegram 线程: 检测到行为配置，正在同步... 目标频道数: {len(target_ids)}")
                    target_map = {"telegram": target_ids}
                    # 更新全局行为引擎
                    global_behavior_engine.update_config(behavior_data, target_map)
                    
                    # 同步到本地 config 对象
                    if isinstance(behavior_data, dict):
                        config.behaviorSettings = BehaviorSettings(**behavior_data)
                    else:
                        config.behaviorSettings = behavior_data
                    config.behaviorTargetChatIds = target_ids

                # --- 步骤 C: 初始化 Client ---
                self.bot_client = TelegramClient()
                self.bot_client.TelegramAgent = config.TelegramAgent
                self.bot_client.memoryLimit = config.memoryLimit
                self.bot_client.separators = config.separators or ["。", "\n", "？", "！"]
                self.bot_client.reasoningVisible = config.reasoningVisible
                self.bot_client.quickRestart = config.quickRestart
                self.bot_client.enableTTS = config.enableTTS
                self.bot_client.wakeWord = config.wakeWord
                self.bot_client.bot_token = config.bot_token
                self.bot_client.config = config
                self.bot_client._manager_ref = weakref.ref(self)
                self.bot_client._ready_callback = self._on_bot_ready

                # --- 步骤 D: 启动行为引擎 (此时 Loop 已在运行，可以 create_task) ---
                if not global_behavior_engine.is_running:
                    asyncio.create_task(global_behavior_engine.start())
                    logging.info("行为引擎已在 Telegram 线程启动")

                # 标记启动完成（允许主线程继续）
                self._startup_complete.set()

                # --- 步骤 E: 运行 Bot (阻塞) ---
                await self.bot_client.run()

            except Exception as e:
                if not self._stop_requested:
                    logging.error(f"Telegram 机器人启动/运行异常: {e}")
                    self._startup_error = str(e)
                # 确保主线程不被卡死
                if not self._startup_complete.is_set():
                    self._startup_complete.set()
                if not self._ready_complete.is_set():
                    self._ready_complete.set()

        # 3. 开始运行 Loop
        try:
            self.loop.run_until_complete(main_startup())
        except Exception as e:
            if not self._stop_requested:
                logging.error(f"Telegram 线程 Loop 异常: {e}")
        finally:
            self._cleanup()
            
    def _on_bot_ready(self):
        """机器人就绪回调（普通函数）"""
        self.is_running = True
        if not self._ready_complete.is_set():
            self._ready_complete.set()
        logging.info("Telegram 机器人已完全就绪")

    def _cleanup(self):
        self.is_running = False
        logging.info("开始清理 Telegram 机器人资源...")
        
        if self.loop and not self.loop.is_closed():
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                
                # Stop loop if running
                if self.loop.is_running():
                    self.loop.stop()
                
                # Close loop
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception as e:
                logging.warning(f"关闭事件循环时出错: {e}")
        
        self.bot_client = None
        self.loop = None
        self._shutdown_event.set()
        logging.info("Telegram 机器人资源清理完成")

    def stop_bot(self):
        if not self.is_running and not self.bot_thread:
            logging.info("Telegram 机器人未在运行")
            return
        
        logging.info("正在停止 Telegram 机器人...")
        self._stop_requested = True
        self.is_running = False
        
        if self.bot_client:
            self.bot_client._shutdown_requested = True
        
        self._shutdown_event.set()
        
        # Increase to 15s (must be > polling timeout)
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=15)
            
            if self.bot_thread.is_alive():
                logging.warning("Telegram 机器人线程未能在15秒内停止")
                # Force cleanup as last resort
                self._cleanup()
        
        self._stop_requested = False
        logging.info("Telegram 机器人停止操作完成")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "thread_alive": self.bot_thread.is_alive() if self.bot_thread else False,
            "client_ready": self.bot_client._is_ready if self.bot_client else False,
            "config": self.config.model_dump() if self.config else None,
            "loop_running": self.loop and not self.loop.is_closed() if self.loop else False,
            "startup_error": self._startup_error,
            "connection_established": self._startup_complete.is_set(),
            "ready_completed": self._ready_complete.is_set(),
            "stop_requested": self._stop_requested,
        }
    
    def update_behavior_config(self, config: TelegramBotConfig):
        """
        热更新行为配置，不重启机器人
        """
        # 更新 Manager 的本地记录
        self.config = config
        
        # 1. 更新 Client 内部的实时参数
        if self.bot_client:
            self.bot_client.TelegramAgent = config.TelegramAgent 
            self.bot_client.enableTTS = config.enableTTS
            self.bot_client.wakeWord = config.wakeWord
            self.bot_client.config = config # 同步整个 config 对象

        # 2. 更新全局行为引擎
        from py.behavior_engine import global_behavior_engine
        target_map = {
            "telegram": config.behaviorTargetChatIds
        }
        
        global_behavior_engine.update_config(
            config.behaviorSettings,
            target_map
        )
        logging.info("Telegram 机器人: 行为配置已热更新，计时器已重置")