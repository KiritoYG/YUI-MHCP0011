import asyncio
import json
import httpx
from typing import Dict, List, Optional, Any
from py.task_center import get_task_center, TaskStatus
from py.get_setting import load_settings, get_port

class SubAgentExecutor:
    """子智能体执行器"""
    
    def __init__(self, workspace_dir: str, settings: Dict):
        self.workspace_dir = workspace_dir
        self.settings = settings
        self.port = get_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.simple_chat_endpoint = f"{self.base_url}/simple_chat"
    
    async def execute_subtask(
        self,
        task_id: str,
        consensus_content: Optional[str] = None,
        max_iterations: int = 30
    ) -> Dict[str, Any]:
        """执行子任务的主循环"""
        task_center = await get_task_center(self.workspace_dir)
        task = await task_center.get_task(task_id)
        
        if not task:
            return {"success": False, "error": f"Task {task_id} not found"}
        
        # 标记任务开始
        await task_center.update_task_progress(
            task_id=task_id, progress=0, status=TaskStatus.RUNNING
        )
        
        iteration = 0
        conversation_history = []
        assistant_only_history = task.context.get("history", [])
        
        system_prompt = self._build_system_prompt(task, consensus_content)
        conversation_history.append({"role": "system", "content": system_prompt})
        
        initial_user_msg = f"请执行以下任务：\n\n{task.description}\n\n要求：完成后请整理出最终结果。"
        conversation_history.append({"role": "user", "content": initial_user_msg})
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as http_client:
                while iteration < max_iterations:
                    iteration += 1
                    current_progress = 10 + int((iteration / max_iterations) * 80)
                    print(f"[SubAgent] Task {task_id} - Iteration {iteration}")
                    
                    # 1. 调用 LLM (流式接收，内部不再传递 flag)
                    assistant_response = await self._call_llm_stream_only(
                        http_client=http_client,
                        messages=conversation_history,
                        model='super-model',
                        task_id=task_id,
                        task_center=task_center,
                        base_progress=current_progress,
                        display_history=assistant_only_history
                    )
                    
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                    # ⭐⭐⭐ 核心逻辑回归：相信数据库状态 ⭐⭐⭐
                    
                    # 2. 重新加载任务状态
                    # 如果刚才执行了 finish_task，这里读出来的就是 COMPLETED
                    latest_task = await task_center.get_task(task_id)
                    
                    if latest_task.status == TaskStatus.COMPLETED:
                        print(f"🚀 [SubAgent] Task {task_id} Status is COMPLETED. Finishing loop.")
                        return {
                            "success": True,
                            "task_id": task_id,
                            "result": latest_task.result,
                            "summary": "任务已完成。",
                            "iterations": iteration
                        }

                    # 3. 只有状态不是 Completed，才继续更新进度和检查隐式完成
                    await task_center.update_task_progress(
                        task_id=task_id,
                        progress=current_progress,
                        status=TaskStatus.RUNNING,
                        context={"history": assistant_only_history, "current_iteration": iteration}
                    )

                    # 4. 隐式完成检查 (没调工具，但在对话里说完成了)
                    is_complete = await self._check_task_completion_smart(
                        task=task,
                        conversation_history=conversation_history,
                        http_client=http_client
                    )
                    
                    if is_complete:
                        print(f"⚡ [SubAgent] Implicit completion detected.")
                        last_response = assistant_response or "任务已完成"
                        
                        await task_center.update_task_progress(
                            task_id=task_id,
                            progress=100,
                            status=TaskStatus.COMPLETED,
                            result=last_response,
                            context={"summary": last_response[:200] + "...", "history": assistant_only_history}
                        )
                        return {
                            "success": True,
                            "task_id": task_id,
                            "result": last_response,
                            "summary": last_response[:200] + "...",
                            "iterations": iteration
                        }
                    
                    conversation_history.append({
                        "role": "user",
                        "content": "请继续执行任务。如果已完成所有步骤，请总结并给出最终结果。"
                    })
                
                # ... 超时处理保持不变 ...
                return {"success": False, "error": "Max iterations reached"}

        except Exception as e:
            # ... 异常处理保持不变 ...
            return {"success": False, "error": str(e)}

    async def _call_llm_stream_only(
        self, 
        http_client: httpx.AsyncClient, 
        messages: List[Dict], 
        model: str,
        task_id: str = None,
        task_center: Any = None,
        base_progress: int = 0,
        display_history: List[str] = None
    ) -> str:
        payload = {
            "messages": messages,
            "model": model,
            "stream": True, 
            "temperature": 0.5,
            "max_tokens": self.settings.get('max_tokens', 4000),
            "is_sub_agent": True,
            "disable_tools": ["create_subtask", "query_tasks_tool", "cancel_subtask"] 
        }

        full_content = ""
        current_text_buffer = ""
        tool_step_counter = 0

        try:
            async with http_client.stream("POST", self.chat_endpoint, json=payload, headers={"Content-Type": "application/json"}) as response:
                if response.status_code != 200:
                    raise Exception(f"API Error {response.status_code}")

                async for line in response.aiter_lines():
                    if not line.strip(): continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]": break
                        try:
                            chunk = json.loads(data_str)
                            if "error" in chunk or not chunk.get("choices"): continue
                            
                            delta = chunk["choices"][0].get("delta", {})

                            # 1. 累积文本
                            content = delta.get("content")
                            if content:
                                full_content += content
                                current_text_buffer += content

                            # 2. 处理工具
                            tool_data = delta.get("tool_content")
                            if tool_data and task_center and task_id:
                                # 刷新缓冲区
                                if current_text_buffer.strip():
                                    display_history.append(current_text_buffer.strip())
                                    current_text_buffer = ""
                                
                                tool_type = tool_data.get("type")
                                tool_title = str(tool_data.get("title", "Unknown")).strip()
                                
                                # ⭐⭐⭐ 关键点：如果是 finish_task，不要碰进度条 ⭐⭐⭐
                                if "finish_task" in tool_title:
                                    # finish_task 工具已经在 task_tools.py 里把状态改成 COMPLETED 了
                                    # 我们千万不能在这里调用 update_task_progress，否则会把状态覆盖回 RUNNING
                                    
                                    # 仅记录到显示历史，不写库
                                    res_content = tool_data.get("content", "")
                                    display_history.append(f"✅ [{tool_title}]\nResult: {str(res_content)[:100]}...")
                                    continue 

                                # 其他普通工具正常更新进度
                                if tool_type in ["tool_result", "error"]:
                                    tool_step_counter += 1
                                    res_content = tool_data.get("content", "")
                                    icon = "✅" if tool_type == "tool_result" else "❌"
                                    
                                    short_res = str(res_content)[:300] + "..." if len(str(res_content)) > 300 else str(res_content)
                                    display_history.append(f"{icon} [{tool_title}]\nResult: {short_res}")
                                    
                                    micro_progress = min(base_progress + (tool_step_counter * 2), 99)
                                    
                                    # 只有非 finish_task 才写入 DB
                                    await task_center.update_task_progress(
                                        task_id=task_id,
                                        progress=micro_progress,
                                        status=TaskStatus.RUNNING,
                                        context={"history": display_history}
                                    )

                        except: continue

        except Exception as e:
             raise Exception(f"Stream Failed: {str(e)}")

        if current_text_buffer.strip() and display_history is not None:
            display_history.append(current_text_buffer.strip())

        return full_content if full_content else "(任务执行中...)"
    
    # ... (其他辅助方法如 _build_system_prompt 等保持不变) ...
    def _build_system_prompt(self, task, consensus_content: Optional[str]) -> str:
        prompt = f"你是一个专业的任务执行助手。\n【任务信息】ID: {task.task_id} | 标题: {task.title}\n【执行要求】专注完成任务，使用可用工具，完成后明确表示结束。"
        if consensus_content: prompt += f"\n\n【共识规范】\n{consensus_content}\n"
        return prompt
    
    async def _check_task_completion_smart(self, task, conversation_history, http_client) -> bool:
        recent = self._get_recent_conversation(conversation_history)
        msgs = [{"role": "system", "content": "判断任务是否完成，只回复YES或NO。"},
                {"role": "user", "content": f"任务：{task.description}\n最近进展：{recent}\n是否完成？"}]
        try:
            resp = await http_client.post(self.simple_chat_endpoint, json={"messages": msgs, "model": "super-model", "stream": False})
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip().upper().startswith("YES")
        except: pass
        return False
    
    async def _extract_final_result(self, task, conversation_history, http_client) -> Dict[str, str]:
        history_str = ""
        for msg in conversation_history:
            if msg["role"] in ["assistant", "user"]:
                content = msg["content"] if msg["content"] else "[执行了工具操作]"
                history_str += f"{msg['role']}: {content}\n"
        msgs = [{"role": "system", "content": "请从对话历史中提取出任务的【最终执行结果】，保留核心干货（如报告内容、代码、分析结果）。"},
                {"role": "user", "content": f"任务目标：{task.description}\n\n对话历史：\n{history_str[-6000:]}\n\n请给出最终结果："}]
        full_res = "未提取到结果"
        try:
            resp = await http_client.post(self.simple_chat_endpoint, json={"messages": msgs, "model": "super-model", "stream": False})
            if resp.status_code == 200: full_res = resp.json()["choices"][0]["message"]["content"].strip()
        except: 
            full_res = "\n".join([m["content"] for m in conversation_history if m["role"] == "assistant" and m["content"]])
        return {"full": full_res, "summary": full_res[:200].replace("\n", " ") + "..."}

    def _get_recent_conversation(self, conversation_history: List[Dict]) -> str:
        texts = []
        for msg in reversed(conversation_history[-5:]):
            texts.append(f"{msg['role']}: {str(msg.get('content'))[:200]}")
        return "\n".join(texts)

async def run_subtask_in_background(task_id: str, workspace_dir: str, settings: Dict, consensus_content: Optional[str] = None):
    executor = SubAgentExecutor(workspace_dir, settings)
    await executor.execute_subtask(task_id, consensus_content)