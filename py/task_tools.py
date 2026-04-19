"""主智能体使用的任务管理工具"""
import asyncio
from typing import Optional
from py.task_center import get_task_center, TaskStatus
from py.sub_agent import run_subtask_in_background

# --- Tool Definitions ---

create_subtask_tool = {
    "type": "function",
    "function": {
        "name": "create_subtask",
        "description": """创建一个子任务并在后台异步执行。

⚠️ 使用场景：
- 将大任务拆分成多个独立的小任务并行执行
- 需要执行耗时较长的任务（如批量处理、深度研究）
- 需要委托给专门的子智能体处理特定领域问题

✅ 特点：
- 异步执行，不阻塞主对话
- 自动保存进度，重启后可恢复
- 可通过 query_task_progress 查看实时状态

📝 返回值：子任务ID，用于后续跟踪进度

⚠️ 注意：
- 每个子任务都是独立的对话上下文
- 子任务无法访问主对话的历史记录（除非在description中明确说明）
- 建议在description中提供完整的背景信息和明确的完成标准
- 如果不是用户要求，子任务创建后请不要主动查询其进度，客户端UI会自动将当前进度和结果显示给用户""",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "子任务的简短标题（建议不超过50字）"
                },
                "description": {
                    "type": "string",
                    "description": """子任务的详细描述，必须包含：
1. 任务目标：期望达成的具体结果
2. 背景信息：必要的上下文和前置知识
3. 完成标准：如何判断任务已完成
4. 约束条件：需要遵守的规则或限制"""
                },
                "agent_type": {
                    "type": "string",
                    "description": "使用的智能体类型（当前固定为 'default'）",
                    "default": "default"
                }
            },
            "required": ["title", "description"]
        }
    }
}

query_tasks_tool = {
    "type": "function",
    "function": {
        "name": "query_task_progress",
        "description": "查询任务进度。支持按任务ID精确查询，或按父任务、状态批量查询。",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": { 
                    "type": "string",
                    "description": "可选：指定任务ID进行精确查询"
                },
                "parent_task_id": {
                    "type": "string",
                    "description": "可选：查询指定父任务下的所有子任务"
                },
                "status": {
                    "type": "string",
                    "description": "可选：过滤特定状态",
                    "enum": ["pending", "running", "completed", "failed", "cancelled"]
                },
                "verbose": {
                    "type": "boolean",
                    "description": "查看已完成任务的完整结果时设为 true",
                    "default": False
                }
            }
        }
    }
}

cancel_subtask_tool = {
    "type": "function",
    "function": {
        "name": "cancel_subtask",
        "description": "取消一个正在执行或待执行的子任务。",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要取消的任务ID"
                }
            },
            "required": ["task_id"]
        }
    }
}

# ⭐ 新增：finish_task_tool
# 注意：此工具应当只提供给子智能体 (SubAgent) 使用
finish_task_tool = {
    "type": "function",
    "function": {
        "name": "finish_task",
        "description": """✅ 任务完成确认工具。
当所有任务目标都已达成时，【必须】调用此工具来正式结束任务。

⚠️ 关键规则：
1. 只有调用此工具，任务状态才会真正变为 COMPLETED。
2. 调用后，请将最终的交付物（代码、报告、结论）放入 result 参数中。
3. 调用此工具后，当前对话流程将立即终止，不要再回复任何额外内容。

❌ 不要仅在对话中说 "我完成了"，必须调用此工具！""",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "当前任务的ID"
                },
                "result": {
                    "type": "string",
                    "description": "最终的任务产出报告。这将作为任务的正式结果展示给父智能体或用户。请确保内容完整、格式清晰（支持Markdown）。"
                }
            },
            "required": ["task_id", "result"]
        }
    }
}

# --- Tool Implementations ---

async def create_subtask(
    title: str,
    description: str,
    agent_type: str = "default",
    workspace_dir: str = None,
    settings: dict = None,
    parent_task_id: Optional[str] = None,
    consensus_content: Optional[str] = None
) -> str:
    """创建并启动子任务"""
    try:
        task_center = await get_task_center(workspace_dir)
        
        # 创建任务
        task = await task_center.create_task(
            title=title,
            description=description,
            parent_task_id=parent_task_id,
            agent_type=agent_type
        )
        
        # 在后台异步执行
        asyncio.create_task(
            run_subtask_in_background(
                task_id=task.task_id,
                workspace_dir=workspace_dir,
                settings=settings, 
                consensus_content=consensus_content
            )
        )
    except Exception as e:
        return f"❌ 创建子任务失败: {str(e)}"
    
    return f"✅ 子任务已创建并开始执行\n\n任务ID: {task.task_id}\n标题: {task.title}\n请不要主动查询任务进度，客户端UI会自动将当前进度和结果显示给用户"

async def query_task_progress(
    workspace_dir: str,
    task_id: Optional[str] = None,  # 新增：接收 task_id
    parent_task_id: Optional[str] = None,
    status: Optional[str] = None,
    verbose: bool = False
) -> str:
    """查询任务进度 - 支持单任务精确查询和列表查询"""
    try:
        from py.task_center import get_task_center, TaskStatus
        
        task_center = await get_task_center(workspace_dir)
        status_enum = TaskStatus(status) if status else None
        
        tasks = []

        # 👉 优化 1：如果有 task_id，优先精确查找，且忽略 status 过滤
        if task_id:
            single_task = await task_center.get_task(task_id)
            if single_task:
                tasks = [single_task]
            else:
                return f"❌ 未找到 ID 为 {task_id} 的任务，请检查 ID 是否正确。"
        
        # 👉 优化 2：如果没有 task_id，再进行列表搜索和过滤
        else:
            status_enum = TaskStatus(status) if status else None
            tasks = await task_center.list_tasks(
                parent_task_id=parent_task_id,
                status=status_enum
            )
        
        if not tasks:
            return "📋 任务中心当前没有相关任务。"
        
        # 构建输出
        result_lines = [f"📋 任务中心状态 (共 {len(tasks)} 个任务)"]
        if verbose:
            result_lines.append("📢 [详情模式] 已开启：正在展示完整结果...")
        result_lines.append("-" * 30)
        
        for task in tasks:
            icon = "✅" if task.status == TaskStatus.COMPLETED else "🔄" if task.status == TaskStatus.RUNNING else "⏳"
            result_lines.append(f"{icon} [{task.task_id}] {task.title}")
            result_lines.append(f"   状态: {task.status.value.upper()} | 进度: {task.progress}%")
            
            history = task.context.get("history", [])
            
            # 运行中
            if task.status == TaskStatus.RUNNING:
                if history:
                    result_lines.append(f"   执行动态: {history[-1][:100]}...")
                if verbose and history:
                    result_lines.append("   📜 已完成步骤:")
                    for i, step in enumerate(history, 1):
                        result_lines.append(f"     {i}. {step[:200]}...")

            # 已完成
            elif task.status == TaskStatus.COMPLETED:
                if verbose:
                    # ✅ 如果 verbose=True，强制显示完整 result
                    result_content = task.result if task.result else "（无结果内容）"
                    result_lines.append(f"   🎯 最终完整产出:\n{result_content}\n")
                    
                    # 可选：显示中间过程
                    if history:
                        result_lines.append("   📜 执行过程回溯 (最近3步):")
                        for i, step in enumerate(history[-3:], 1):
                            result_lines.append(f"     ... {step[:100]} ...")
                else:
                    summary = task.context.get('summary') or (task.result[:150] + "..." if task.result else "无结果内容")
                    result_lines.append(f"   📝 结果摘要: {summary}")
                    result_lines.append(f"   💡 (提示: 使用 verbose=true 可查看完整报告)")

            elif task.status == TaskStatus.FAILED:
                result_lines.append(f"   ❌ 错误信息: {task.error}")

            result_lines.append("") 
    except Exception as e:
        return f"❌ 查询任务进度失败: {str(e)}"

    return "\n".join(result_lines)

async def cancel_subtask(workspace_dir: str, task_id: str) -> str:
    """取消子任务"""
    try:
        task_center = await get_task_center(workspace_dir)
        success = await task_center.cancel_task(task_id)
    except Exception as e:
        return f"❌ 取消任务失败: {str(e)}"
    return f"✅ 任务 {task_id} 已取消" if success else f"❌ 取消任务 {task_id} 失败"

# ⭐ 新增实现：finish_task
async def finish_task(
    workspace_dir: str,
    task_id: str,
    result: str
) -> str:
    try:
        """子智能体调用此函数来标记任务完成"""
        task_center = await get_task_center(workspace_dir)
        
        # 强制更新为 COMPLETED，进度 100，并保存最终结果
        success = await task_center.update_task_progress(
            task_id=task_id,
            progress=100,
            status=TaskStatus.COMPLETED,
            result=result
        )
    except Exception as e:
        return f"❌ 标记任务完成失败: {str(e)}"
    
    if success:
        return f"🎉 任务 {task_id} 已成功标记为完成！结果已保存。请停止后续操作。"
    else:
        return f"❌ 任务 {task_id} 状态更新失败（可能任务ID错误）。"