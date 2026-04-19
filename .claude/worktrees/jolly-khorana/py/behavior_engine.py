import asyncio
import time
import datetime
import logging
import random
from typing import Dict, List, Callable, Optional, Any, Union
from pydantic import BaseModel

# --- 数据模型定义 (与前端一致) ---

class BehaviorTriggerTime(BaseModel):
    timeValue: str  # "HH:mm:ss"
    days: List[int] = [] # 1=Mon...6=Sat, 0=Sun

class BehaviorTriggerNoInput(BaseModel):
    latency: int

class BehaviorTriggerCycle(BaseModel):
    cycleValue: str # "HH:mm:ss"
    repeatNumber: int
    isInfiniteLoop: bool

class BehaviorTrigger(BaseModel):
    type: str  # "time", "noInput", "cycle"
    time: Optional[BehaviorTriggerTime] = None
    noInput: Optional[BehaviorTriggerNoInput] = None
    cycle: Optional[BehaviorTriggerCycle] = None

class BehaviorRandomAction(BaseModel):
    events: List[str]
    type: str # "random", "order"
    orderIndex: int = 0

class BehaviorAction(BaseModel):
    type: str # "prompt", "random", "topic"
    prompt: Optional[str] = ""
    random: Optional[BehaviorRandomAction] = None
    topicLimit: int = 1

class BehaviorItem(BaseModel):
    enabled: bool
    trigger: BehaviorTrigger
    action: BehaviorAction
    platform: str # "chat", "feishu", "dingtalk", "all"

class BehaviorSettings(BaseModel):
    enabled: bool
    behaviorList: List[BehaviorItem] = []

# --- 通用行为引擎 ---

class BehaviorEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BehaviorEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        
        self.settings: Optional[BehaviorSettings] = None
        self.is_running = False
        self._stop_event = None # 延迟初始化
        self.platform_activity: Dict[str, Dict[str, float]] = {} 
        self.platform_targets: Dict[str, List[str]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}

    def register_handler(self, platform: str, handler: Callable):
        """注册平台的执行回调函数"""
        self.handlers[platform] = handler
        if platform not in self.platform_activity:
            self.platform_activity[platform] = {}
            
        # 关键修复：当新平台注册时，如果已经有配置，重置计时器
        # 这样即使“先开设置再开机器人”，机器人一上线就会重新计算触发时间
        if self.settings and self.settings.enabled:
            self.timers.clear()
            self.counters.clear()
            logging.info(f"[BehaviorEngine] 平台 {platform} 已上线，重置引擎计时器以激活任务")
        
        print(f"[BehaviorEngine] 已注册平台: {handler}")

    def update_config(self, settings: Union[BehaviorSettings, dict], platform_targets: Dict[str, List[str]] = None):
        """热更新配置"""
        if isinstance(settings, dict):
            try:
                self.settings = BehaviorSettings(**settings)
            except Exception as e:
                logging.error(f"[BehaviorEngine] 配置解析失败: {e}")
                return
        else:
            self.settings = settings

        if platform_targets:
            for platform, targets in platform_targets.items():
                self.platform_targets[platform] = targets
            
        self.timers.clear()
        self.counters.clear()
        logging.info("[BehaviorEngine] 配置已更新，计时器已重置")

    def report_activity(self, platform: str, chat_id: str):
        """平台层调用：上报活跃状态（重置无输入计时）"""
        if platform not in self.platform_activity:
            self.platform_activity[platform] = {}
        self.platform_activity[platform][chat_id] = time.time()

    async def start(self):
        """启动引擎循环"""
        # 确保 Event 对象在当前的 Loop 中创建
        self._stop_event = asyncio.Event()
        self.is_running = True
        logging.info("[BehaviorEngine] 监控任务已激活")
        
        try:
            while not self._stop_event.is_set():
                if not self.is_running: 
                    break
                try:
                    await self._tick()
                except Exception as e:
                    logging.error(f"[BehaviorEngine] Tick 异常: {e}")
                
                # 必须使用 asyncio.sleep，不能用 time.sleep
                await asyncio.sleep(1)
        finally:
            self.is_running = False
            logging.info("[BehaviorEngine] 监控循环已安全退出")

    def stop(self):
        """停止引擎"""
        self.is_running = False
        if self._stop_event:
            self._stop_event.set()
        logging.info("[BehaviorEngine] 已发出停止信号")

    async def _tick(self):
        """核心逻辑：每秒检查一次"""
        if not self.settings or not self.settings.enabled:
            return

        now = time.time()
        dt_now = datetime.datetime.now()

        current_time_str = dt_now.strftime("%H:%M") 
        py_weekday = dt_now.weekday()
        current_day = (py_weekday + 1) if py_weekday < 6 else 0

        for idx, behavior in enumerate(self.settings.behaviorList):
            if not behavior.enabled: continue
            
            # 确定当前行为要分发到哪些平台
            target_platforms = []
            if behavior.platform == "all":
                target_platforms = list(self.handlers.keys())
            elif behavior.platform in self.handlers:
                target_platforms = [behavior.platform]
            
            for platform in target_platforms:
                handler = self.handlers.get(platform)
                if not handler: continue

                trigger_chats = []
                static_targets = self.platform_targets.get(platform, [])

                # --- 逻辑 1: 无输入 (No Input) ---
                if behavior.trigger.type == "noInput" and behavior.trigger.noInput:
                    latency = behavior.trigger.noInput.latency
                    active_targets = list(self.platform_activity.get(platform, {}).keys())
                    for chat_id in active_targets:
                        last_active = self.platform_activity[platform].get(chat_id, now)
                        if now - last_active >= latency:
                            uniq_key = f"noInput_{idx}_{platform}_{chat_id}"
                            if self.timers.get(uniq_key, 0) < now - latency - 5: # 防抖
                                trigger_chats.append(chat_id)
                                self.timers[uniq_key] = now

                # --- 逻辑 2: 定时 (Time) ---
                elif behavior.trigger.type == "time" and behavior.trigger.time:
                    # 前端传的是 "HH:mm:ss"，我们只比对 "HH:mm"
                    if behavior.trigger.time.timeValue.startswith(current_time_str):
                        if not behavior.trigger.time.days or current_day in behavior.trigger.time.days:
                            uniq_key = f"time_{idx}_{platform}_{current_time_str}"
                            if self.timers.get(uniq_key, 0) < now - 65:
                                trigger_chats = static_targets
                                self.timers[uniq_key] = now

                # --- 逻辑 3: 周期 (Cycle) ---
                elif behavior.trigger.type == "cycle" and behavior.trigger.cycle:
                    try:
                        t = behavior.trigger.cycle.cycleValue.split(':')
                        cycle_sec = int(t[0])*3600 + int(t[1])*60 + int(t[2])
                    except: cycle_sec = 60
                    
                    uniq_key = f"cycle_{idx}_{platform}"
                    if self.timers.get(uniq_key, 0) == 0: # 首次运行
                        self.timers[uniq_key] = now + cycle_sec
                    elif now >= self.timers.get(uniq_key, 0):
                        count_key = f"cycle_count_{idx}_{platform}"
                        count = self.counters.get(count_key, 0)
                        if behavior.trigger.cycle.isInfiniteLoop or count < behavior.trigger.cycle.repeatNumber:
                            trigger_chats = static_targets
                            self.timers[uniq_key] = now + cycle_sec
                            self.counters[count_key] = count + 1

                # 执行触发
                if trigger_chats:
                    for chat_id in set(trigger_chats):
                        if chat_id:
                            logging.info(f"[BehaviorEngine] 命中规则 {idx}，准备推送到 {platform}:{chat_id}")
                            asyncio.create_task(handler(chat_id, behavior))

# 全局单例
global_behavior_engine = BehaviorEngine()