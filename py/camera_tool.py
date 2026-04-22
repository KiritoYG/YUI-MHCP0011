import asyncio
import base64
import io
from typing import Optional

# OpenCV for camera capture
OPENCV_AVAILABLE = False
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("⚠️ [Warning] opencv-python 未安装，摄像头功能不可用。请运行: pip install opencv-python")

# pywin32 for window capture (Windows only)
WIN32_AVAILABLE = False
try:
    import win32gui
    import win32ui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    pass


def _frame_to_jpeg_bytes(frame_bgr) -> bytes:
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _capture_camera_bytes_sync(camera_index: int) -> bytes:
    if not OPENCV_AVAILABLE:
        raise RuntimeError("opencv-python 未安装，请运行: pip install opencv-python")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}，请检查设备连接")
    try:
        # 读几帧让摄像头曝光稳定
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"摄像头 {camera_index} 读取失败")
        return _frame_to_jpeg_bytes(frame)
    finally:
        cap.release()


def _capture_camera_sync(camera_index: int) -> str:
    return base64.b64encode(_capture_camera_bytes_sync(camera_index)).decode('utf-8')


def _capture_window_bytes_sync(window_title: str) -> bytes:
    if not WIN32_AVAILABLE:
        raise RuntimeError("pywin32 不可用，无法捕获指定窗口（非Windows系统或未安装pywin32）")

    hwnd = None
    def enum_handler(h, _):
        nonlocal hwnd
        title = win32gui.GetWindowText(h)
        if title and window_title.lower() in title.lower():
            hwnd = h
    win32gui.EnumWindows(enum_handler, None)

    if hwnd is None:
        raise RuntimeError(f"找不到标题包含 '{window_title}' 的窗口，请确认程序已运行且未最小化")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        raise RuntimeError(f"窗口 '{window_title}' 尺寸异常（{w}x{h}），可能已最小化")

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitmap = win32ui.CreateBitmap()
    saveBitmap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitmap)
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

    bmpinfo = saveBitmap.GetInfo()
    bmpstr = saveBitmap.GetBitmapBits(True)

    win32gui.DeleteObject(saveBitmap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    from PIL import Image
    img = Image.frombuffer(
        'RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1
    )
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _capture_window_sync(window_title: str) -> str:
    return base64.b64encode(_capture_window_bytes_sync(window_title)).decode('utf-8')


# ==================== 按 HWND 直接截窗口 ====================
def _capture_hwnd_bytes_sync(hwnd: int) -> bytes:
    if not WIN32_AVAILABLE:
        raise RuntimeError("pywin32 不可用")
    if not win32gui.IsWindow(hwnd):
        raise RuntimeError(f"无效窗口句柄 {hwnd}")
    if win32gui.IsIconic(hwnd):
        # 最小化：尝试还原（若失败就放弃）
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        except Exception:
            pass

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        raise RuntimeError(f"窗口尺寸异常（{w}x{h}）")

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitmap = win32ui.CreateBitmap()
    saveBitmap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitmap)
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

    bmpinfo = saveBitmap.GetInfo()
    bmpstr = saveBitmap.GetBitmapBits(True)

    win32gui.DeleteObject(saveBitmap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    from PIL import Image
    img = Image.frombuffer(
        'RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1
    )
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


# ==================== 显示器截图（支持多屏 / 全部） ====================
def _list_monitors_sync() -> list:
    """列出所有显示器（Windows）。返回 [{index, name, x, y, width, height, is_primary}, ...]"""
    monitors = []
    if not WIN32_AVAILABLE:
        # 回退：用 PIL 取主屏
        try:
            from PIL import ImageGrab
            im = ImageGrab.grab()
            monitors.append({"index": 0, "name": "Primary", "x": 0, "y": 0,
                             "width": im.width, "height": im.height, "is_primary": True})
        except Exception:
            pass
        return monitors
    try:
        import win32api
        for i, m in enumerate(win32api.EnumDisplayMonitors()):
            try:
                info = win32api.GetMonitorInfo(m[0])
                l, t, r, b = info['Monitor']
                monitors.append({
                    "index": i,
                    "name": info.get('Device', f'Monitor {i+1}'),
                    "x": int(l), "y": int(t),
                    "width": int(r - l), "height": int(b - t),
                    "is_primary": bool(info.get('Flags', 0) & 1),
                })
            except Exception:
                continue
    except Exception:
        pass
    return monitors


def _capture_monitor_bytes_sync(monitor_index: int = 0) -> bytes:
    """截取指定显示器（monitor_index=-1 表示所有屏拼接）。"""
    from PIL import ImageGrab
    if monitor_index < 0:
        img = ImageGrab.grab(all_screens=True)
    else:
        mons = _list_monitors_sync()
        if not mons:
            img = ImageGrab.grab()
        else:
            if monitor_index >= len(mons):
                monitor_index = 0
            m = mons[monitor_index]
            bbox = (m['x'], m['y'], m['x'] + m['width'], m['y'] + m['height'])
            img = ImageGrab.grab(bbox=bbox, all_screens=True)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=85)
    return buf.getvalue()


# ==================== 枚举窗口 / 摄像头（供前端下拉） ====================
def _list_windows_sync() -> list:
    """列出可见的顶层窗口，供 UI 下拉选择。"""
    results = []
    if not WIN32_AVAILABLE:
        return results
    try:
        import win32process
    except Exception:
        win32process = None

    psutil = None
    try:
        import psutil as _psutil
        psutil = _psutil
    except Exception:
        pass

    def cb(h, _):
        try:
            if not win32gui.IsWindowVisible(h):
                return
            title = win32gui.GetWindowText(h)
            if not title or len(title.strip()) == 0:
                return
            l, t, r, b = win32gui.GetWindowRect(h)
            w, hh = r - l, b - t
            if w < 100 or hh < 80:
                return
            pid = 0
            if win32process is not None:
                try:
                    _, pid = win32process.GetWindowThreadProcessId(h)
                except Exception:
                    pid = 0
            exe = ""
            if psutil is not None and pid:
                try:
                    exe = psutil.Process(pid).name()
                except Exception:
                    exe = ""
            results.append({
                "hwnd": int(h), "title": title, "pid": int(pid),
                "exe": exe, "width": int(w), "height": int(hh),
            })
        except Exception:
            pass

    try:
        win32gui.EnumWindows(cb, None)
    except Exception:
        pass
    # 去掉一些明显的系统外壳窗口
    IGNORE_EXE = {"explorer.exe", "textinputhost.exe", "searchhost.exe", "shellexperiencehost.exe",
                  "applicationframehost.exe", "startmenuexperiencehost.exe"}
    results = [w for w in results if (w["exe"] or "").lower() not in IGNORE_EXE or w["exe"] == ""]
    results.sort(key=lambda x: (-x["width"] * x["height"], x["title"].lower()))
    return results


def _list_cameras_sync(max_probe: int = 5) -> list:
    """探测摄像头。每探一个要打开设备一次，代价略高（~0.2-0.5s/台），前端应按需调用。"""
    results = []
    if not OPENCV_AVAILABLE:
        return results
    for i in range(max_probe):
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _frm = cap.read()
                if ok:
                    results.append({"index": i, "name": f"Camera {i}"})
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return results


# ==================== 统一捕获入口 ====================
def _capture_source_bytes_sync(
    source_type: str = "camera",
    index: Optional[int] = None,
    hwnd: Optional[int] = None,
    window_title: Optional[str] = None,
) -> bytes:
    """
    source_type:
      - "camera":  摄像头 (index=摄像头索引)
      - "monitor": 显示器 (index=显示器索引; -1=全部屏拼接)
      - "window":  窗口   (hwnd 优先；否则用 window_title 模糊匹配)
    """
    st = (source_type or "camera").lower()
    if st == "monitor":
        return _capture_monitor_bytes_sync(int(index) if index is not None else 0)
    if st == "window":
        if hwnd:
            return _capture_hwnd_bytes_sync(int(hwnd))
        if window_title:
            return _capture_window_bytes_sync(window_title)
        raise ValueError("source_type=window 需要 hwnd 或 window_title")
    # 默认 camera
    return _capture_camera_bytes_sync(int(index) if index is not None else 0)


async def capture_source_bytes(
    source_type: str = "camera",
    index: Optional[int] = None,
    hwnd: Optional[int] = None,
    window_title: Optional[str] = None,
) -> bytes:
    return await asyncio.to_thread(
        _capture_source_bytes_sync, source_type, index, hwnd, window_title
    )


async def list_capture_sources() -> dict:
    """并发枚举所有可选捕获源。"""
    monitors, windows, cameras = await asyncio.gather(
        asyncio.to_thread(_list_monitors_sync),
        asyncio.to_thread(_list_windows_sync),
        asyncio.to_thread(_list_cameras_sync),
    )
    return {"monitors": monitors, "windows": windows, "cameras": cameras}


async def capture_snapshot_bytes(
    camera_index: int = 0,
    window_title: Optional[str] = None,
    source_type: Optional[str] = None,
    index: Optional[int] = None,
    hwnd: Optional[int] = None,
) -> bytes:
    """
    一次性捕获一帧，返回原始 JPEG 字节。
    旧参数 camera_index/window_title 仍兼容；新参数 source_type/index/hwnd 更通用。
    """
    if source_type:
        return await capture_source_bytes(source_type, index=index, hwnd=hwnd, window_title=window_title)
    if window_title:
        return await asyncio.to_thread(_capture_window_bytes_sync, window_title)
    return await asyncio.to_thread(_capture_camera_bytes_sync, camera_index)


# 给 Qwen3-VL 的客观描述指令 —— 要求输出中立事实，避免带主观情绪/评价，
# 这样才不会污染主模型（DeepSeek）的人格输出。
_OBJECTIVE_SYSTEM_PROMPT = (
    "你是一个视觉识别引擎，只负责客观、简洁、结构化地描述画面内容。"
    "禁止使用第一人称、禁止表达情感或喜好、禁止主观评价（如'温馨''漂亮''可爱'等），"
    "也不要向用户说话。只输出视觉事实。"
)

_DEFAULT_FACTUAL_PROMPT = (
    "请客观列出画面中可见的事实信息：\n"
    "1. 人物：数量、性别表征、衣着、表情、动作、相对位置\n"
    "2. 场景：地点类型、环境光线、主要物体\n"
    "3. 文字：所有可读的文字内容（原样抄录）\n"
    "4. 其他显著细节\n"
    "若画面中有明显的说话特征（张嘴、手势、聊天气泡/名牌高亮等），请单独指出。"
)


async def camera_capture(
    prompt: Optional[str] = None,
    camera_index: int = 0,
    window_title: Optional[str] = None,
    source_type: Optional[str] = None,
    index: Optional[int] = None,
    hwnd: Optional[int] = None,
) -> str:
    """
    从摄像头或指定游戏窗口（如VRChat）截图，通过视觉模型（Qwen3-VL）生成客观视觉描述。
    返回的是**视觉参考数据**，主模型应据此以自身人格向用户回复，而非直接复述。

    Args:
        prompt: 给视觉模型的具体问题。留空则使用设置里的默认指令
        camera_index: 摄像头索引（0=默认摄像头，1=第二个...）
        window_title: 要捕获的窗口标题关键词（如 "VRChat"），优先于摄像头
    """
    from py.get_setting import load_settings
    from openai import AsyncOpenAI

    settings = await load_settings()
    vision_cfg = settings.get('vision', {})

    if not vision_cfg.get('enabled'):
        return (
            "[视觉工具不可用] 视觉模型未启用。请告知用户：在设置 → 视觉模型 中开启并配置 "
            "Base URL (如 http://localhost:8001/v1)、模型名 (如 Qwen/Qwen3-VL-7B-Instruct)、"
            "API Key (vLLM 填 EMPTY)。"
        )

    # 确定实际使用的 prompt：调用方指定 > 设置中默认 > 硬编码事实性默认
    effective_prompt = (
        prompt
        or vision_cfg.get('cameraPrompt')
        or _DEFAULT_FACTUAL_PROMPT
    )

    # 若调用方没传 source_type，回退到设置里的默认源（GUI 里选的那个）
    if source_type is None:
        source_type = vision_cfg.get('captureSourceType')  # 'camera' | 'monitor' | 'window' | None
        if index is None:
            index = vision_cfg.get('captureSourceIndex')
        if hwnd is None:
            hwnd = vision_cfg.get('captureWindowHwnd')
        if not window_title:
            window_title = vision_cfg.get('captureWindowTitle')

    # 捕获图像
    try:
        if source_type in ('camera', 'monitor', 'window'):
            raw_bytes = await capture_source_bytes(
                source_type,
                index=index if index is not None else (camera_index if source_type == 'camera' else 0),
                hwnd=hwnd,
                window_title=window_title,
            )
            b64 = base64.b64encode(raw_bytes).decode('utf-8')
            if source_type == 'monitor':
                source = f"显示器 #{index if index is not None else 0}"
            elif source_type == 'window':
                source = f"窗口「{window_title or f'hwnd={hwnd}'}」"
            else:
                source = f"摄像头 #{index if index is not None else camera_index}"
        elif window_title:
            b64 = await asyncio.to_thread(_capture_window_sync, window_title)
            source = f"窗口「{window_title}」"
        else:
            b64 = await asyncio.to_thread(_capture_camera_sync, camera_index)
            source = f"摄像头 #{camera_index}"
    except Exception as e:
        return f"[视觉工具失败] 图像捕获失败：{e}。请告知用户检查设备连接或窗口是否运行。"

    # 调用视觉模型 —— 优先用摄像头专属配置，回退到通用 vision 配置
    try:
        api_key = vision_cfg.get('cameraApiKey') or vision_cfg.get('api_key') or 'EMPTY'
        # 默认走本地 Qwen VL 服务（8001），避免和 faster-qwentts（8000）撞端口
        base_url = vision_cfg.get('cameraBaseUrl') or vision_cfg.get('base_url') or 'http://localhost:8001/v1'
        model = vision_cfg.get('cameraModel') or vision_cfg.get('model') or 'Qwen3-VL'
        # 视觉识别用低温度保证客观性，不跟随通用温度设置
        temperature = 0.2

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _OBJECTIVE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": effective_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            temperature=temperature,
        )
        raw = (response.choices[0].message.content or "").strip()

        # 用明确的"参考信息"框架包裹，防止主模型把视觉引擎的原话当成自己的回答复读
        return (
            f"[视觉参考信息 | 来源: {source}]\n"
            f"{raw}\n"
            f"[以上为视觉引擎的客观识别结果，请你结合自身人格与上下文，"
            f"用你自己的语气向用户描述/回应，不要照搬以上文字。]"
        )
    except Exception as e:
        return (
            f"[视觉工具失败] 视觉模型调用失败：{e}。"
            "请告知用户检查：vLLM 服务是否已启动、Base URL / 模型名是否正确。"
        )


# ================= Tool Schema =================

camera_capture_tool = {
    "type": "function",
    "function": {
        "name": "camera_capture",
        "description": (
            "【你的眼睛】调用此工具即可看到外置摄像头画面或指定游戏窗口（如VRChat）的实时截图。"
            "本工具内部会把图像交给本地视觉引擎（Qwen3-VL），返回一段客观的视觉事实描述。\n\n"
            "使用场景：用户让你「看看」「瞅一眼」「摄像头里是什么」「VRC里谁在说话」"
            "「帮我看下房间」「识别屏幕上的文字」等任何需要视觉感知的请求。\n\n"
            "重要：返回值是视觉引擎的机械识别结果，不是你的回复。"
            "你必须基于这些事实信息，用自己的人格与语气向用户表达，不要逐字复读返回的描述。"
            "例如返回里说「画面中有一只橘色猫咪趴在床上」，你应该以自己的风格说"
            "（示例）「哎，看到啦～你家那只橘猫又在床上瘫着呢」。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "【可选】给视觉引擎的具体问题。留空则走设置里的默认指令（全面事实描述）。"
                        "当用户有明确关注点时再传入，例如："
                        "'识别正在说话的人物并描述其头像特征与位置'、"
                        "'抄录屏幕上所有可读文字'、"
                        "'列出桌面上的所有物品'。"
                        "注意：这是给视觉引擎的命令，不是给用户的话——用命令式、客观中立的措辞。"
                    )
                },
                "source_type": {
                    "type": "string",
                    "enum": ["camera", "monitor", "window"],
                    "description": (
                        "【可选】捕获源类型：'camera'=摄像头、'monitor'=显示器/桌面、'window'=指定窗口。"
                        "不传则使用用户在设置里选的默认源。"
                        "用户说「看桌面」→monitor；「看摄像头」→camera；「看游戏/VRChat/那个窗口」→window。"
                    )
                },
                "index": {
                    "type": "integer",
                    "description": (
                        "【可选】当 source_type=camera 时为摄像头索引，=monitor 时为显示器索引（-1=全屏拼接）。"
                        "默认 0。"
                    )
                },
                "camera_index": {
                    "type": "integer",
                    "description": "【旧参数，兼容保留】摄像头索引。等价于 source_type='camera' 时的 index。",
                    "default": 0
                },
                "window_title": {
                    "type": "string",
                    "description": (
                        "【可选】要截图的窗口标题关键词（部分匹配），如 'VRChat'、'Chrome'。"
                        "指定后等价于 source_type='window'。"
                    )
                },
                "hwnd": {
                    "type": "integer",
                    "description": "【可选】窗口句柄，精确指定某个窗口。通常由前端在枚举后传入。"
                }
            },
            "required": []
        }
    }
}

camera_use_tools = [camera_capture_tool]
