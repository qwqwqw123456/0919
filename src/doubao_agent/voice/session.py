from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .vad import EnergyVAD, SAMPLE_RATE

_SENTENCE_END = re.compile(r"[。！？!?；;\n]")


@dataclass
class VoiceEvent:
    kind: str                      # speech_start | speech_end | user_text | reply_text | audio | barge_in | error
    t_ms: float
    text: str = ""
    audio: bytes = b""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "t_ms": round(self.t_ms, 1)}
        if self.text:
            d["text"] = self.text
        if self.meta:
            d["meta"] = self.meta
        return d


def split_sentences(text: str) -> list[str]:
    """按句切分，短于 4 字的碎片并到下一句，避免 TTS 一句一个词。"""
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        buf += p
        if len(buf) >= 4:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out or ([text.strip()] if text.strip() else [])


class VoiceSession:
    """一轮语音通话的双工状态机。

    职责：VAD 断句 → ASR 出文本 → 调回复函数拿答案 → 逐句 TTS → 播报；
    播报中检测到人声立刻打断（barge_in），清掉未播的句子，并把已播文本记进上下文。
    真不真、快不快由引擎决定；这里只管"谁在说话、什么时候该停"。
    """

    def __init__(
        self,
        asr,
        tts,
        reply: Callable[[str, list[dict[str, str]]], Awaitable[str]],
        *,
        vad: EnergyVAD | None = None,
        max_sentences: int = 6,
        sentence_chars: int = 60,
        quick_ack: str = "嗯，我在听。",
        tts_workers: int = 3,
        ack_tts=None,
        ack_audio: tuple[bytes, int] | None = None,
    ) -> None:
        self.asr = asr
        self.tts = tts
        self.reply = reply            # async (user_text, history) -> str
        self.vad = vad or EnergyVAD(sample_rate=SAMPLE_RATE)
        self.max_sentences = max_sentences
        self.sentence_chars = sentence_chars
        self.history: list[dict[str, str]] = []
        self._t0 = time.monotonic()
        self._pending_audio: list[bytes] = []
        self._spoken: list[str] = []
        self.quick_ack = quick_ack
        self.ack_tts = ack_tts or tts   # 垫话优先用本地快引擎，别让主音色的网络延迟拖首声
        self._ack_pcm: bytes | None = ack_audio[0] if ack_audio else None
        self._ack_rate = ack_audio[1] if ack_audio else 16000
        self._pool = ThreadPoolExecutor(max_workers=max(1, tts_workers), thread_name_prefix="tts")

    def now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000

    def mark_start(self) -> None:
        """把计时零点挪到"真正开始收话"这一刻：模型/音色预热不算进通话延迟。"""
        self._t0 = time.monotonic()

    @property
    def playing(self) -> bool:
        return bool(self._pending_audio)

    def push_audio(self, pcm: bytes) -> list[VoiceEvent]:
        """喂音频，返回即时事件（开口/打断/收尾）。收尾后需 await respond()。"""
        events: list[VoiceEvent] = []
        for ev in self.vad.feed(pcm):
            if ev.kind == "speech_start":
                if self.playing:
                    dropped = len(self._pending_audio)
                    self._pending_audio.clear()
                    self._spoken_append("（被打断，未播完）")
                    events.append(
                        VoiceEvent("barge_in", self.now_ms(), meta={"dropped_chunks": dropped})
                    )
                events.append(VoiceEvent("speech_start", self.now_ms()))
            elif ev.kind == "speech_end":
                events.append(
                    VoiceEvent("speech_end", self.now_ms(), audio=ev.audio,
                               meta={"bytes": len(ev.audio), "seconds": round(len(ev.audio) / 2 / SAMPLE_RATE, 2)})
                )
        return events

    def _spoken_append(self, text: str) -> None:
        if text:
            self._spoken.append(text)

    async def respond(self, user_audio: bytes) -> list[VoiceEvent]:
        """对一段完整用户语音给出回复（汇总版：调用方一次性拿走全部事件）。"""
        return [ev async for ev in self.stream_audio(user_audio)]

    async def respond_to_text(self, text: str) -> list[VoiceEvent]:
        """跳过 ASR（对方的话已经是文字时的入口，也用于端到端自测）。"""
        return [ev async for ev in self.stream_text(text)]

    async def stream_audio(self, user_audio: bytes):
        """流式版：识别→垫话→作答→逐句音频，一有就先吐出去。"""
        try:
            asr_out = self.asr.transcribe(user_audio)
        except Exception as exc:  # noqa: BLE001 - 识别失败也要让通话继续，不炸会话
            yield VoiceEvent("error", self.now_ms(), text=f"ASR 失败：{type(exc).__name__}: {exc}")
            return
        text = str(asr_out.get("text") or "").strip()
        if not text:
            yield VoiceEvent("error", self.now_ms(), text="没听清（识别结果为空）")
            return
        async for ev in self._answer_stream(text, asr_out):
            yield ev

    async def stream_text(self, text: str):
        text = str(text or "").strip()
        if not text:
            yield VoiceEvent("error", self.now_ms(), text="输入为空")
            return
        async for ev in self._answer_stream(text, {"engine": "given"}):
            yield ev

    async def _answer(self, text: str, asr_meta: dict[str, Any]) -> list[VoiceEvent]:
        return [ev async for ev in self._answer_stream(text, asr_meta)]

    async def _answer_stream(self, text: str, asr_meta: dict[str, Any]):
        """逐句边合成边吐事件：第一句不等后面几句，通话能早 1–2 秒出声。"""
        yield VoiceEvent("user_text", self.now_ms(), text=text, meta=asr_meta)
        self.history.append({"role": "user", "content": text})
        # 先垫一句极短应答：模型还在想的时候，对方先听到人声（电话里最忌死寂）
        if self.quick_ack:
            try:
                if self._ack_pcm is None:
                    self._ack_pcm, self._ack_rate = self.ack_tts.synthesize(self.quick_ack)
                ack_pcm, ack_rate = self._ack_pcm, self._ack_rate
                self._pending_audio.append(ack_pcm)
                yield VoiceEvent("ack", self.now_ms(), text=self.quick_ack, audio=ack_pcm,
                                 meta={"rate": ack_rate, "bytes": len(ack_pcm),
                                       "seconds": round(len(ack_pcm) / 2 / ack_rate, 2)})
            except Exception as exc:  # noqa: BLE001 - 垫话失败不影响正题
                yield VoiceEvent("error", self.now_ms(), text=f"垫话合成失败：{exc}")
        try:
            answer = await self.reply(text, self.history)
        except Exception as exc:  # noqa: BLE001
            yield VoiceEvent("error", self.now_ms(), text=f"回复生成失败：{exc}")
            return
        answer = str(answer or "").strip() or "我在，你继续说。"
        yield VoiceEvent("reply_text", self.now_ms(), text=answer)
        self.history.append({"role": "assistant", "content": answer})

        pieces = [s[: self.sentence_chars] for s in split_sentences(answer)[: self.max_sentences]]
        if not pieces:
            return
        loop = asyncio.get_running_loop()
        pool = self._pool
        first = await loop.run_in_executor(pool, self.tts.synthesize, pieces[0])
        rest = (
            asyncio.gather(*[loop.run_in_executor(pool, self.tts.synthesize, p) for p in pieces[1:]])
            if len(pieces) > 1 else None
        )
        for piece in pieces[:1]:
            yield self._audio_event(piece, first)
        if rest is not None:
            for piece, (pcm, rate) in zip(pieces[1:], await rest):
                yield self._audio_event(piece, (pcm, rate))

    def _audio_event(self, piece: str, audio: tuple[bytes, int]) -> VoiceEvent:
        pcm, rate = audio
        self._pending_audio.append(pcm)
        self._spoken_append(piece)
        return VoiceEvent("audio", self.now_ms(), text=piece, audio=pcm,
                          meta={"rate": rate, "bytes": len(pcm), "seconds": round(len(pcm) / 2 / rate, 2)})

    def ack_payload(self) -> tuple[bytes, int] | None:
        """已合成好的垫话，供上层跨会话复用（省掉每次冷启的合成开销）。"""
        return (self._ack_pcm, self._ack_rate) if self._ack_pcm else None

    def prewarm(self) -> None:
        """会话开始就把垫话合成好，真说话时首声接近 0ms。"""
        if not self.quick_ack or self._ack_pcm is not None:
            return
        try:
            self._ack_pcm, self._ack_rate = self.ack_tts.synthesize(self.quick_ack)
        except Exception:  # noqa: BLE001 - 预热失败就当没有垫话
            self._ack_pcm = b""

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def drain_audio(self) -> list[bytes]:
        """取走待播音频（调用方播放后即视为已播）。"""
        out, self._pending_audio = self._pending_audio, []
        return out

    def transcript(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)
