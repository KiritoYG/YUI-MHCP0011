import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import aiofiles
import aiofiles.os
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SubTask(BaseModel):
    task_id: str
    parent_task_id: Optional[str] = None
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0  # 0-100
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    agent_type: str = "default"
    # 使用 Field(default_factory=dict) 确保每个实例有独立的字典，防止引用污染
    context: Dict[str, Any] = Field(default_factory=dict)

class TaskCenter:
    """任务中心 - 管理所有主任务和子任务"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.task_dir = self.workspace_dir / ".agent" / "tasks"
        self._lock = asyncio.Lock()
        self._ensure_task_dir()
    
    def _ensure_task_dir(self):
        """确保任务目录存在"""
        self.task_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_task_file(self, task_id: str) -> Path:
        """获取任务文件路径"""
        return self.task_dir / f"{task_id}.json"
    
    async def create_task(
        self,
        title: str,
        description: str,
        parent_task_id: Optional[str] = None,
        agent_type: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> SubTask:
        """创建新任务"""
        async with self._lock:
            task_id = str(uuid.uuid4())[:8]
            now = datetime.now().isoformat()
            
            task = SubTask(
                task_id=task_id,
                parent_task_id=parent_task_id,
                title=title,
                description=description,
                created_at=now,
                updated_at=now,
                agent_type=agent_type,
                context=context or {}
            )
            
            await self._save_task(task)
            return task
    
    async def _save_task(self, task: SubTask):
        """保存任务到文件"""
        task_file = self._get_task_file(task.task_id)
        async with aiofiles.open(task_file, 'w', encoding='utf-8') as f:
            await f.write(task.model_dump_json(indent=2))
    
    async def get_task(self, task_id: str) -> Optional[SubTask]:
        """获取任务详情"""
        task_file = self._get_task_file(task_id)
        if not task_file.exists():
            return None
        
        try:
            async with aiofiles.open(task_file, 'r', encoding='utf-8') as f:
                data = await f.read()
                return SubTask.model_validate_json(data)
        except Exception as e:
            print(f"Error loading task {task_id}: {e}")
            return None
    
    async def update_task_progress(
        self,
        task_id: str,
        progress: int,
        status: Optional[TaskStatus] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新任务进度和上下文"""
        async with self._lock:
            task = await self.get_task(task_id)
            if not task:
                return False
            
            # --- 核心修复：进度计算逻辑优化 ---
            
            # 1. 基础范围限制 (0-100)
            safe_progress = max(0, min(100, progress))
            
            # 2. 确定目标状态
            target_status = status if status else task.status
            
            if target_status == TaskStatus.COMPLETED:
                # 策略A：如果任务完成，强制进度为 100%
                final_progress = 100
            elif target_status == TaskStatus.FAILED:
                 # 策略B：如果失败，保持当前最大进度或设定的进度，但不强制100
                final_progress = max(task.progress, safe_progress)
            elif target_status == TaskStatus.CANCELLED:
                # 策略C：取消任务通常归零或保持现状，这里选择保持现状以免丢失上下文
                final_progress = task.progress 
            else:
                # 策略D：运行中 (PENDING/RUNNING)
                # 规则1：单调递增，不许回退 (取 old 和 new 的最大值)
                final_progress = max(task.progress, safe_progress)
                # 规则2：运行中封顶 99%，防止未完成却显示 100% 误导用户
                final_progress = min(99, final_progress)
            
            task.progress = final_progress
            task.updated_at = datetime.now().isoformat()
            
            # 3. 更新状态和时间戳
            if status:
                task.status = status
                if status == TaskStatus.RUNNING and not task.started_at:
                    task.started_at = datetime.now().isoformat()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.now().isoformat()
            
            if result is not None:
                task.result = result
            
            if error is not None:
                task.error = error
                task.status = TaskStatus.FAILED

            # 4. 合并上下文数据 (增量更新)
            if context is not None:
                task.context.update(context)
            
            await self._save_task(task)
            return True

    async def list_tasks(
        self,
        parent_task_id: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> List[SubTask]:
        """列出任务"""
        tasks = []
        
        if not self.task_dir.exists():
            return tasks
        
        # 获取所有json文件
        files = list(self.task_dir.glob("*.json"))
        
        for task_file in files:
            try:
                async with aiofiles.open(task_file, 'r', encoding='utf-8') as f:
                    data = await f.read()
                    task = SubTask.model_validate_json(data)
                    
                    if parent_task_id is not None and task.parent_task_id != parent_task_id:
                        continue
                    if status is not None and task.status != status:
                        continue
                    
                    tasks.append(task)
            except Exception as e:
                print(f"Error loading task file {task_file}: {e}")
                continue
        
        # 按创建时间倒序排序
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 取消时将状态设为 CANCELLED，进度通常不再增加
        return await self.update_task_progress(
            task_id=task_id,
            progress=0, # 这里的数值会被上面的逻辑覆盖为 keep current
            status=TaskStatus.CANCELLED
        )

    async def delete_task(self, task_id: str) -> bool:
        """删除任务文件"""
        async with self._lock:
            task_file = self._get_task_file(task_id)
            if task_file.exists():
                try:
                    await aiofiles.os.remove(task_file)
                    return True
                except Exception as e:
                    print(f"Error deleting task {task_id}: {e}")
                    return False
            return False

    async def cleanup_old_tasks(self, days: int = 7):
        """清理旧任务（待实现）"""
        pass

# --- 全局任务中心实例管理 ---

# 全局任务中心实例字典 {workspace_path: TaskCenter}
_task_centers: Dict[str, TaskCenter] = {}

async def get_task_center(workspace_dir: str) -> TaskCenter:
    """获取或创建任务中心实例"""
    if workspace_dir not in _task_centers:
        _task_centers[workspace_dir] = TaskCenter(workspace_dir)
    return _task_centers[workspace_dir]