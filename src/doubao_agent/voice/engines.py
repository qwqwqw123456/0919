from __future__ import annotations

import io
import time
import wave
from array import array
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000


def resample_to_16k(samples: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE or len(samples) == 0:
        return samples.astype(np.float32)
    n = int(len(samples) * SAMPLE_RATE / src_rate)
    idx = np.linspace(0, len(samples) - 1, n)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)


def pcm16_from_wav(path: str | Path, target_rate: int = SAMPLE_RATE) -> bytes:
    """读 wav → 16k/16bit/mono 的 PCM 字节，喂给 VAD 或 ASR。"""
    with wave.open(str(path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"只支持 16bit wav，实际 {width * 8}bit")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    a = resample_to_16k(a, rate)
    return (a * 32767).astype(np.int16).tobytes()


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def decode_audio_any(data: bytes | str | Path, target_rate: int = SAMPLE_RATE) -> bytes:
    """任意常见音频（mp3/wav/m4a/ogg）→ 16k/16bit/mono PCM；用 PyAV，避免依赖 ffmpeg 命令行。"""
    import av  # 延迟导入

    src = data if isinstance(data, (str, Path)) else io.BytesIO(data)
    with av.open(src) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        rate = target_rate
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                buf = rf.to_ndarray()
                chunks.append(buf.reshape(-1).astype(np.int16))
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray().reshape(-1).astype(np.int16))
    if not chunks:
        raise ValueError("解码后没有音频数据")
    return np.concatenate(chunks).astype(np.int16).tobytes()


class MockASR:
    """测试用 ASR：不识别内容，按固定文本返回，用来验证链路与状态机。"""

    name = "mock"

    def __init__(self, text: str = "帮我盯着明天的机票，降价了就提醒我") -> None:
        self.text = text

    def transcribe(self, pcm: bytes, sr: int = SAMPLE_RATE) -> dict[str, Any]:
        return {
            "text": self.text,
            "seconds": round(len(pcm) / 2 / sr, 2),
            "engine_ms": 1,
            "engine": self.name,
        }


def _run_coro(make_coro: Any) -> Any:
    """在当前线程跑协程；若本线程已有事件循环（如 CLI 直接同步调合成），放独立线程跑。

    asyncio.run 在运行中的 loop 内会直接抛错，之前被 except 吞掉，白退子进程还留协程告警。
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-loop") as ex:
        return ex.submit(lambda: asyncio.run(make_coro())).result()


class MockTTS:
    """测试用 TTS：按字数产生正弦音，长度可控，便于验证分句与打断。"""

    name = "mock"
    rate = SAMPLE_RATE

    def __init__(self, per_char_ms: int = 120, freq: float = 440.0) -> None:
        self.per_char_ms = per_char_ms
        self.freq = freq

    def synthesize(self, text: str) -> tuple[bytes, int]:
        n = int(self.rate * self.per_char_ms / 1000 * max(1, len(text)))
        t = np.arange(n) / self.rate
        wave_ = 0.3 * np.sin(2 * np.pi * self.freq * t)
        return (wave_ * 32767).astype(np.int16).tobytes(), self.rate


class WhisperASR:
    """faster-whisper 本地识别：CPU int8，base 模型下整句 ≈ 0.3–0.8s。"""

    name = "whisper"

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8") -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # 延迟导入：没装引擎也能跑 mock

            self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        return self._model

    def transcribe(self, pcm: bytes, sr: int = SAMPLE_RATE) -> dict[str, Any]:
        model = self._ensure()
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        a = resample_to_16k(a, sr)
        t0 = time.time()
        segs, info = model.transcribe(a, language="zh", beam_size=1)
        text = "".join(s.text for s in list(segs)).strip()  # 必须先消费，否则计时不准
        return {
            "text": text,
            "seconds": round(len(a) / SAMPLE_RATE, 2),
            "engine_ms": int((time.time() - t0) * 1000),
            "engine": self.name,
            "lang_prob": round(float(getattr(info, "language_probability", 0.0)), 2),
        }


class EdgeTTS:
    """Edge 在线 TTS（zh-CN-XiaoxiaoNeural）：音色自然，无需密钥；无网时退回 Piper。

    走子进程调官方 CLI，避免在已有事件循环里再开 loop。
    """

    name = "edge"

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        timeout: float = 20.0,
        fallback: Any = None,
    ) -> None:
        self.voice = voice
        self.timeout = timeout
        self.fallback = fallback  # 无网/接口变更时退本地 Piper

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """优先走 edge_tts 的异步 API（~0.5–1s），失败再退回子进程（~2s），最后退 fallback。"""
        try:
            import edge_tts

            async def _run() -> bytes:
                chunks: list[bytes] = []
                async for chunk in edge_tts.Communicate(text, self.voice).stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
                return b"".join(chunks)

            mp3 = _run_coro(_run)
            if mp3:
                return decode_audio_any(mp3, SAMPLE_RATE), SAMPLE_RATE
        except Exception:  # noqa: BLE001 - 网络/版本问题一律退回子进程
            pass

        import subprocess
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
            out = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "edge_tts", "--voice", self.voice, "--text", text, "--write-media", out],
                capture_output=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-300:])
            pcm = decode_audio_any(Path(out), SAMPLE_RATE)
            return pcm, SAMPLE_RATE
        except Exception:  # noqa: BLE001
            if self.fallback is None:
                raise
            return self.fallback.synthesize(text)
        finally:
            Path(out).unlink(missing_ok=True)


class PiperTTS:
    """Piper 中文语音合成（huayan-medium）。输出统一转 16k/16bit/mono。"""

    name = "piper"

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).expanduser()
        self._voice: Any = None

    def _ensure(self) -> Any:
        if self._voice is None:
            from piper import PiperVoice  # 延迟导入

            self._voice = PiperVoice.load(str(self.model_path), config_path=str(self.model_path) + ".json")
        return self._voice

    def synthesize(self, text: str) -> tuple[bytes, int]:
        voice = self._ensure()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize_wav(text, w)
        buf.seek(0)
        with wave.open(buf, "rb") as r:
            rate = r.getframerate()
            raw = r.readframes(r.getnframes())
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        a = resample_to_16k(a, rate)
        return (a * 32767).astype(np.int16).tobytes(), SAMPLE_RATE


def build_engines(cfg: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """按配置选引擎；缺模型/缺依赖时退 mock，并把原因写进 errors 由调用方决定是否降级。"""
    cfg = dict(cfg or {})
    asr_kind = str(cfg.get("asr") or "whisper").lower()
    tts_kind = str(cfg.get("tts") or "piper").lower()

    if asr_kind == "whisper":
        asr = WhisperASR(
            model_size=str(cfg.get("whisper_model") or "base"),
            compute_type=str(cfg.get("whisper_compute") or "int8"),
        )
    else:
        asr = MockASR()

    model_path = str(cfg.get("piper_model") or Path.home() / "tools/voice/zh_CN-huayan-medium.onnx")
    if tts_kind == "edge":
        local = PiperTTS(model_path) if Path(model_path).expanduser().is_file() else None
        tts = EdgeTTS(str(cfg.get("edge_voice") or "zh-CN-XiaoxiaoNeural"), fallback=local)
    elif tts_kind == "piper" and Path(model_path).expanduser().is_file():
        tts = PiperTTS(model_path)
    else:
        tts = MockTTS()
    return asr, tts


__all__ = [
    "SAMPLE_RATE",
    "MockASR",
    "EdgeTTS",
    "MockTTS",
    "PiperTTS",
    "WhisperASR",
    "build_engines",
    "pcm16_from_wav",
    "resample_to_16k",
    "wav_bytes",
]
