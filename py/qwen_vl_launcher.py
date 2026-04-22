"""
Qwen3.5-VL 本地模型一键启动器

基于子进程托管 vLLM（或任意 OpenAI 兼容推理服务）来提供本地视觉模型。
对外暴露 start/stop/status 三个异步接口，供 server.py 路由层调用。

设计要点：
- 单实例：同一时刻只允许一个托管进程
- 非阻塞：启动后立即返回，通过后台线程读 stdout 做健康日志
- 端口可用探测：以 TCP 探测 /v1/models 判断是否就绪
- 默认命令：vllm serve <model_path> --host 127.0.0.1 --port <port> --served-model-name <name>
  用户可在 settings 中覆盖完整命令模板
"""

from __future__ import annotations

import asyncio
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Optional, List, Dict, Any

import httpx


def _create_kill_on_close_job():
    """
    创建一个 Windows Job Object，带 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE。
    当本 Python 进程死亡（含硬杀），Job 句柄被关闭，OS 自动把 Job 内所有子进程一起杀掉。
    返回 Job 句柄；非 Windows / 不支持则返回 None。
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def _assign_process_to_job(job, pid: int) -> bool:
    if not job or os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        hproc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not hproc:
            return False
        try:
            ok = kernel32.AssignProcessToJobObject(job, hproc)
            return bool(ok)
        finally:
            kernel32.CloseHandle(hproc)
    except Exception:
        return False


class QwenVLLauncher:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log_buffer: deque[str] = deque(maxlen=400)
        self._start_time: float = 0.0
        self._config_snapshot: Dict[str, Any] = {}
        self._reader_thread: Optional[threading.Thread] = None
        # 进程级 Job Object：父进程死亡时 OS 自动杀掉所有 Job 内子进程
        self._job = _create_kill_on_close_job()

    # ----------------------------- 内部工具 -----------------------------
    def _port_in_use(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except Exception:
            return False

    def _append_log(self, line: str) -> None:
        try:
            self._log_buffer.append(line.rstrip())
        except Exception:
            pass

    def _reader_loop(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    line = str(raw)
                self._append_log(line)
        except Exception as e:
            self._append_log(f"[reader exited] {e}")

    def _build_cmd(
        self,
        model_path: str,
        host: str,
        port: int,
        served_name: str,
        extra_args: str,
        command_template: Optional[str],
    ) -> List[str]:
        # 可自定义模板；占位符 {model} {host} {port} {name} {extra}
        if command_template:
            tpl = command_template.format(
                model=model_path,
                host=host,
                port=port,
                name=served_name,
                extra=extra_args or "",
            )
            return shlex.split(tpl, posix=False)

        # 默认启动自带的 transformers OpenAI 兼容服务（Windows 友好，不依赖 vLLM）
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_vl_server_transformers.py")
        cmd = [
            sys.executable,
            server_script,
            "--model-path",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--served-name",
            served_name,
        ]
        # extra_args 里原本给 vLLM 的参数（如 --dtype / --max-model-len / --limit-mm-per-prompt）
        # 与 transformers 后端不完全兼容，我们只挑能识别的：--dtype / --max-new-tokens
        if extra_args:
            tokens = shlex.split(extra_args, posix=False)
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t in ("--dtype", "--max-new-tokens") and i + 1 < len(tokens):
                    cmd.extend([t, tokens[i + 1]])
                    i += 2
                else:
                    # 其它参数静默忽略（避免 transformers server 报 unknown argument）
                    i += 1 if not t.startswith("--") else (2 if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else 1)
        return cmd

    # ----------------------------- 对外 API -----------------------------
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    async def status(self, host: str = "127.0.0.1", port: int = 8000) -> Dict[str, Any]:
        running = self.is_running()
        port_alive = self._port_in_use(host, port)
        ready = False
        model_list: List[str] = []
        if port_alive:
            try:
                async with httpx.AsyncClient(timeout=2.0) as c:
                    r = await c.get(f"http://{host}:{port}/v1/models")
                    if r.status_code == 200:
                        ready = True
                        data = r.json()
                        model_list = [m.get("id") for m in data.get("data", []) if m.get("id")]
            except Exception:
                pass

        pid = None
        try:
            if self._proc is not None:
                pid = self._proc.pid
        except Exception:
            pid = None

        return {
            "running": running,
            "port_alive": port_alive,
            "ready": ready,
            "pid": pid,
            "uptime": (time.time() - self._start_time) if running else 0,
            "models": model_list,
            "config": self._config_snapshot,
            "logs_tail": list(self._log_buffer)[-50:],
        }

    async def start(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        served_name: str = "Qwen3.5-VL",
        extra_args: str = "",
        command_template: Optional[str] = None,
        wait_ready: bool = False,
        ready_timeout: float = 90.0,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"ok": False, "error": "already_running", "pid": self._proc.pid}
            if not model_path or not os.path.exists(model_path):
                return {"ok": False, "error": f"model_path not found: {model_path}"}
            if self._port_in_use(host, port):
                return {"ok": False, "error": f"port {port} already in use"}

            cmd = self._build_cmd(model_path, host, port, served_name, extra_args, command_template)
            self._log_buffer.clear()
            self._append_log("[launcher] cmd: " + " ".join(cmd))

            creationflags = 0
            if os.name == "nt":
                # 脱离控制台，便于整棵进程树统一管理
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    bufsize=0,
                )
            except FileNotFoundError as e:
                return {"ok": False, "error": f"launch failed: {e}"}
            except Exception as e:
                return {"ok": False, "error": f"launch failed: {e}"}

            self._proc = proc
            self._start_time = time.time()
            if self._job is not None:
                if _assign_process_to_job(self._job, proc.pid):
                    self._append_log("[launcher] assigned to job object (kill-on-close)")
                else:
                    self._append_log("[launcher] warning: failed to assign to job object")
            self._config_snapshot = {
                "model_path": model_path,
                "host": host,
                "port": port,
                "served_name": served_name,
                "extra_args": extra_args,
                "command_template": command_template,
            }
            t = threading.Thread(target=self._reader_loop, args=(proc,), daemon=True)
            t.start()
            self._reader_thread = t

        # 可选：等待就绪
        if wait_ready:
            deadline = time.time() + ready_timeout
            while time.time() < deadline:
                st = await self.status(host=host, port=port)
                if st.get("ready"):
                    return {"ok": True, "pid": proc.pid, "ready": True, "status": st}
                if not self.is_running():
                    return {
                        "ok": False,
                        "error": "process exited before ready",
                        "logs_tail": list(self._log_buffer)[-50:],
                    }
                await asyncio.sleep(1.5)
            return {"ok": True, "pid": proc.pid, "ready": False, "warning": "timeout waiting ready"}

        return {"ok": True, "pid": proc.pid, "ready": False}

    async def stop(self, force_timeout: float = 15.0) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._proc = None
                return {"ok": True, "note": "not running"}

            try:
                if os.name == "nt":
                    # 给进程组发 Ctrl-Break（对 vLLM 比 terminate 更干净）
                    try:
                        proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                    except Exception:
                        proc.terminate()
                else:
                    proc.terminate()
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

        # 等待退出
        deadline = time.time() + force_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            await asyncio.sleep(0.3)
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

        with self._lock:
            self._proc = None
        self._append_log("[launcher] stopped")
        return {"ok": True}


# 全局单例
qwen_vl_launcher = QwenVLLauncher()
