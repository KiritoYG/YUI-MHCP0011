# sherpa_asr.py
import os
import asyncio
from pathlib import Path
from io import BytesIO
from py.get_setting import DEFAULT_ASR_DIR
import platform

# ---------- 占位符与全局变量 ----------
_recognizer = None
_last_model_name = None
_last_language = None

# ---------- 懒加载工具函数 ----------
def _detect_device() -> str:
    """延迟检测最佳推理设备"""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            return 'cuda'
    except Exception:
        pass

    return 'cpu'

def _get_recognizer(model_name: str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue", language: str = "auto"):
    """初始化/获取识别器（包含重型库的懒加载）"""
    global _recognizer, _last_model_name, _last_language

    # 如果已经加载且模型和语言都没变，直接返回
    if _recognizer is not None and model_name == _last_model_name and language == _last_language:
        print(f"[Sherpa] 复用已有识别器 (model={model_name}, language={language})")
        return _recognizer

    # 语言有变化，强制销毁旧识别器
    if _recognizer is not None:
        print(f"[Sherpa] 语言从 '{_last_language}' 切换到 '{language}'，重建识别器...")
        _recognizer = None

    # --- 延迟导入重型依赖 ---
    try:
        import sherpa_onnx
    except ImportError as e:
        print("未安装 sherpa_onnx 库:",e)
        return None

    model_dir = Path(DEFAULT_ASR_DIR) / model_name
    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"

    # 检查文件是否存在
    if not model_path.is_file() or not tokens_path.is_file():
        print(f"提示: Sherpa 模型文件尚未下载，ASR 功能暂不可用。路径: {model_dir}")
        return None

    device = _detect_device()

    # 语言映射:
    # "auto"  -> "" (SenseVoice 自动检测)
    # "zh"    -> "zh" (强制中文)
    # "ja"    -> "ja" (强制日语)
    # "en"    -> "en" (强制英语)
    # "zh+ja" -> "" (自动检测，中日都能识别)
    if language in ("auto", "zh+ja", ""):
        sense_voice_lang = ""
    else:
        sense_voice_lang = language

    # 仅中文时启用 ITN（逆文本正规化），其他语言关闭避免输出被破坏
    enable_itn = (language in ("auto", "zh", "zh+ja", ""))

    print(f"[Sherpa] 正在加载模型 [{model_name}]")
    print(f"[Sherpa] 设备={device}, 请求语言={language}, SenseVoice language='{sense_voice_lang}', ITN={enable_itn}")

    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_path),
            tokens=str(tokens_path),
            num_threads=4,
            provider=device,
            use_itn=enable_itn,
            debug=False,
            language=sense_voice_lang,
        )
        _recognizer = recognizer
        _last_model_name = model_name
        _last_language = language
        print(f"[Sherpa] 识别器创建成功! language='{sense_voice_lang}'")
        return _recognizer
    except TypeError as e:
        # 可能是旧版 sherpa_onnx 不支持 language 参数，尝试不带 language 创建
        print(f"[Sherpa] 带 language 参数创建失败 ({e})，尝试不带 language 参数...")
        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_path),
                num_threads=4,
                provider=device,
                use_itn=True,
                debug=False,
            )
            _recognizer = recognizer
            _last_model_name = model_name
            _last_language = language
            print(f"[Sherpa] 识别器创建成功 (无 language 参数, 旧版兼容模式)")
            return _recognizer
        except Exception as e2:
            print(f"[Sherpa] 加载模型彻底失败: {e2}")
            return None
    except Exception as e:
        print(f"[Sherpa] 加载模型时发生错误: {e}")
        return None

# ---------- 核心同步逻辑 (运行在线程池中) ----------
def _process_audio_sync(recognizer, audio_bytes: bytes) -> str:
    """
    同步执行的 CPU 密集型任务：解码音频 + 神经网络推理
    """
    import soundfile as sf
    import numpy as np

    with BytesIO(audio_bytes) as audio_file:
        audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
        audio = audio[:, 0] # 转单声道

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        recognizer.decode_stream(stream)
        return stream.result.text

# ---------- 公开的异步接口 ----------
async def sherpa_recognize(audio_bytes: bytes, model_name: str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue", language: str = "auto"):
    """
    异步封装：将繁重的推理任务扔到线程池
    language: "auto" | "zh" | "ja" | "zh+ja" | "en" | "ko" | "yue"
    """
    try:
        print(f"[Sherpa] sherpa_recognize 被调用, language={language}")
        recognizer = _get_recognizer(model_name, language)
        if recognizer is None:
            raise RuntimeError("ASR 模型未就绪（可能未下载或加载失败）")

        text = await asyncio.to_thread(_process_audio_sync, recognizer, audio_bytes)
        print(f"[Sherpa] 识别完成: '{text}'")
        return text
    except Exception as e:
        raise RuntimeError(f"Sherpa ASR 处理失败: {e}")
