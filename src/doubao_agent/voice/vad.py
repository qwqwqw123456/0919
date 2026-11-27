from __future__ import annotations

import math
from array import array
from dataclasses import dataclass

SAMPLE_RATE = 16000


@dataclass
class VadEvent:
    kind: str          # speech_start | speech_end
    t_ms: float
    audio: bytes = b""  # 仅 speech_end 带整段 PCM（16k/16bit/mono）


class EnergyVAD:
    """流式能量 VAD：够快、够稳，不引第三方模型。

    判据：短时 RMS 超过阈值且持续 min_speech_ms 才算开口；静音持续 hangover_ms 才算说完。
    阈值自适应：启动时按噪声底噪抬 12dB，避免安静房间里太小、嘈杂环境里太敏感。
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 20,
        min_speech_ms: int = 200,
        hangover_ms: int = 600,
        floor_multiplier: float = 4.0,
        max_utterance_ms: int = 30_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms
        self.floor_multiplier = floor_multiplier
        self.max_utterance_ms = max_utterance_ms

        self._frame_samples = sample_rate * frame_ms // 1000
        self._pending = array("h")
        self._elapsed_ms = 0.0
        self._noise_floor = 200.0        # 初始 RMS 底噪（16bit 量级）
        self._calibrated_frames = 0
        self._speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance = bytearray()

    @property
    def speaking(self) -> bool:
        return self._speaking

    def _rms(self, block: array) -> float:
        if not block:
            return 0.0
        return math.sqrt(sum(float(x) * x for x in block) / len(block))

    def feed(self, pcm: bytes) -> list[VadEvent]:
        """喂一段任意长度的 PCM16；返回这一批里发生的开口/收尾事件。"""
        self._pending.frombytes(pcm)
        events: list[VadEvent] = []
        while len(self._pending) >= self._frame_samples:
            block = self._pending[: self._frame_samples]
            del self._pending[: self._frame_samples]
            events.extend(self._step(block))
        return events

    def flush(self) -> list[VadEvent]:
        """会话结束时把未闭合的一段收掉。"""
        if self._speaking and self._utterance:
            self._speaking = False
            ev = VadEvent("speech_end", self._elapsed_ms, bytes(self._utterance))
            self._utterance = bytearray()
            return [ev]
        return []

    def _step(self, block: array) -> list[VadEvent]:
        self._elapsed_ms += self.frame_ms
        rms = self._rms(block)
        if self._calibrated_frames < 25:  # 前 500ms 学底噪
            self._noise_floor = max(80.0, 0.8 * self._noise_floor + 0.2 * rms)
            self._calibrated_frames += 1
        threshold = self._noise_floor * self.floor_multiplier
        loud = rms > threshold

        if not self._speaking:
            if loud:
                self._speech_ms += self.frame_ms
                self._utterance.extend(block.tobytes())
                if self._speech_ms >= self.min_speech_ms:
                    self._speaking = True
                    self._silence_ms = 0
                    return [VadEvent("speech_start", self._elapsed_ms)]
            else:
                self._speech_ms = 0
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
                self._utterance = bytearray()
            return []

        # 已在说话：继续攒音频，静音够久就收尾
        self._utterance.extend(block.tobytes())
        if loud:
            self._silence_ms = 0
        else:
            self._silence_ms += self.frame_ms
        if self._silence_ms >= self.hangover_ms or len(self._utterance) >= self.max_utterance_ms * 32:
            self._speaking = False
            audio = bytes(self._utterance)
            self._utterance = bytearray()
            self._speech_ms = 0
            self._silence_ms = 0
            return [VadEvent("speech_end", self._elapsed_ms, audio)]
        return []
