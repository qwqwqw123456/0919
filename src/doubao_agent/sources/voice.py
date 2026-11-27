from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec
from ..voice.engines import PiperTTS, build_engines, decode_audio_any, wav_bytes
from ..voice.session import VoiceSession

VOICE_DIR = Path.home() / "tools" / "voice"


class VoiceSource:
    """语音通道：把"听一句、答一句、说回去"做成体系可调用的能力。

    本轮先解决双工链路（VAD 断句、打断、逐句合成），不碰运营商外呼：
    外呼要线路资质与实名，属于另一层，接不接由用户决定。
    """

    name = "voice"
    kind = SourceKind.BUILTIN

    def __init__(
        self,
        reply: Callable[[str], Any],
        engines_cfg: dict[str, Any] | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self.reply = reply
        self._takes_history = True
        self.cfg = dict(engines_cfg or {})
        self.out_dir = Path(out_dir or Path.home() / "agent-state" / "voice")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._asr = None
        self._tts = None

    def bind_reply(self, reply: Callable[[str], Any], takes_history: bool = True) -> None:
        """构造顺序问题：Runtime 实例晚于本通道创建，创建后把真作答函数回填进来。"""
        self.reply = reply
        self._takes_history = takes_history

    def _engines(self):
        if self._asr is None or self._tts is None:
            self._asr, self._tts = build_engines(self.cfg)
            model = Path(str(self.cfg.get("piper_model") or VOICE_DIR / "zh_CN-huayan-medium.onnx")).expanduser()
            self._ack_tts = self._tts if type(self._tts).__name__ == "PiperTTS" else (
                PiperTTS(model) if model.is_file() else self._tts
            )
        return self._asr, self._tts

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="transcribe",
                description=(
                    "把一段音频文件转成文字（wav/mp3/m4a/ogg 均可，内部统一重采样到 16k/16bit/mono）。"
                    "会议片段、通话录音、短视频口白都可以直接丢进来。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "音频文件绝对路径"},
                        "engine": {"type": "string", "description": "whisper 或 mock，默认取配置", "default": "whisper"},
                    },
                    "required": ["path"],
                },
                kind=self.kind,
                risk=Risk.READ,
                idempotent=True,
                source=self.name,
            ),
            ToolSpec(
                name="say",
                description=(
                    "把一句话合成语音并落成 wav，返回文件路径与时长。"
                    "默认用本地 Piper 中文音色（huayan-medium），不联网、不计费。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "要念的文字"},
                        "filename": {"type": "string", "description": "可选，输出文件名（不含目录）"},
                    },
                    "required": ["text"],
                },
                kind=self.kind,
                risk=Risk.REVERSIBLE,
                idempotent=False,
                source=self.name,
            ),
            ToolSpec(
                name="call",
                description=(
                    "跑完一轮语音通话：读一段用户语音 → VAD 断句 → 转写 → 模型作答 → 逐句合成 → "
                    "写入对话音频与时间线。用于验证双工链路、也用于把「打电话那一轮」留痕。"
                    "返回回复文本、回复音频路径、各阶段耗时（ASR/首句 TTS/总时长）。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "用户语音文件（wav，16bit）"},
                        "text": {"type": "string", "description": "可选：不给音频时直接用这段文字当作对方说的话"},
                    },
                    "required": [],
                },
                kind=self.kind,
                risk=Risk.REVERSIBLE,
                idempotent=False,
                source=self.name,
            ),
        ]

    def make_session(self) -> VoiceSession:
        """给实时双工端点复用：同一套引擎与作答函数，含垫话预热。"""
        asr, tts = self._engines()
        session = VoiceSession(
            asr, tts, lambda text, history: self._reply(text, history), ack_tts=self._ack_tts
        )
        session.prewarm()
        session.mark_start()
        return session

    async def call_tool(self, call: ToolCall) -> ToolResult:
        args = call.arguments or {}
        if call.name == "transcribe":
            path = str(args.get("path") or "").strip()
            if not path or not Path(path).is_file():
                return ToolResult(call_id=call.call_id, ok=False, error=f"音频不存在：{path or '（空）'}")
            try:
                pcm = decode_audio_any(path)
                asr, _ = self._engines()
                out = asr.transcribe(pcm)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(call_id=call.call_id, ok=False, error=f"{type(exc).__name__}: {exc}")
            return ToolResult(
                call_id=call.call_id,
                ok=True,
                content=f"{out['text']}\n（{out['seconds']}s，引擎 {out['engine']}，耗时 {out['engine_ms']}ms）",
                meta=out,
            )

        if call.name == "say":
            text = str(args.get("text") or "").strip()
            if not text:
                return ToolResult(call_id=call.call_id, ok=False, error="text 不能为空")
            try:
                _, tts = self._engines()
                pcm, rate = tts.synthesize(text)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(call_id=call.call_id, ok=False, error=f"TTS 失败：{exc}")
            name = str(args.get("filename") or f"say-{int(time.time())}.wav")
            if not name.endswith(".wav"):
                name += ".wav"
            out_path = self.out_dir / Path(name).name
            out_path.write_bytes(wav_bytes(pcm, rate))
            return ToolResult(
                call_id=call.call_id,
                ok=True,
                content=f"{out_path}\n（{len(pcm) / 2 / rate:.2f}s，{rate}Hz）",
                meta={"path": str(out_path), "seconds": round(len(pcm) / 2 / rate, 2), "rate": rate},
            )

        if call.name == "call":
            return await self._call(call, args)

        return ToolResult(call_id=call.call_id, ok=False, error=f"unknown voice tool: {call.name}")

    async def _call(self, call: ToolCall, args: dict[str, Any]) -> ToolResult:
        asr, tts = self._engines()
        session = VoiceSession(
            asr, tts, lambda text, history: self._reply(text, history), ack_tts=self._ack_tts
        )
        t_prewarm = time.time()
        session.prewarm()
        warmup_ms = round((time.time() - t_prewarm) * 1000, 1)
        session.mark_start()
        timeline: list[dict[str, Any]] = []
        t_start = time.time()

        path = str(args.get("path") or "").strip()
        given_text = str(args.get("text") or "").strip()
        if not path and not given_text:
            return ToolResult(call_id=call.call_id, ok=False, error="给 path 或 text 之一")
        if path:
            if not Path(path).is_file():
                return ToolResult(call_id=call.call_id, ok=False, error=f"音频不存在：{path}")
            try:
                pcm = decode_audio_any(path)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(call_id=call.call_id, ok=False, error=f"读音频失败：{exc}")
            # 按 20ms 帧喂，模拟实时流：验证断句时机与打断，而不是整段直塞
            ends: list[bytes] = []
            for i in range(0, len(pcm), 640):
                for ev in session.push_audio(pcm[i : i + 640]):
                    timeline.append(ev.to_dict())
                    if ev.kind == "speech_end" and ev.audio:
                        ends.append(ev.audio)
            for ev in session.vad.flush():
                timeline.append(ev.to_dict())
                if ev.audio:
                    ends.append(ev.audio)
            utterance = ends[-1] if ends else pcm
            events = await session.respond(utterance)
        else:
            timeline.append({"kind": "user_text", "t_ms": 0, "text": given_text, "meta": {"engine": "given"}})
            events = await session.respond_to_text(given_text)

        timeline += [e.to_dict() for e in events]

        audio_blob = b"".join(e.audio for e in events if e.kind == "audio")
        out_path = self.out_dir / f"call-reply-{int(t_start)}.wav"
        if audio_blob:
            out_path.write_bytes(wav_bytes(audio_blob, 16000))

        reply_text = "".join(e.text for e in events if e.kind == "reply_text")
        asr_ms = next((e.meta.get("engine_ms") for e in events if e.kind == "user_text"), None)
        first_audio_ms = next((e.t_ms for e in events if e.kind == "audio"), None)
        timeline.append({"kind": "done", "t_ms": round(session.now_ms(), 1), "warmup_ms": warmup_ms})

        ok = bool(reply_text)
        return ToolResult(
            call_id=call.call_id,
            ok=ok,
            content=(
                f"对方：{next((e.text for e in events if e.kind == 'user_text'), '')}\n"
                f"回复：{reply_text}\n音频：{out_path if audio_blob else '（无音频）'}\n"
                f"ASR {asr_ms}ms / 首句合成 {first_audio_ms}ms / 总 {timeline[-1]['t_ms']}ms"
            ),
            error=None if ok else "未生成回复",
            meta={"reply": reply_text, "audio": str(out_path) if audio_blob else "",
                  "timeline": timeline, "asr_ms": asr_ms, "first_audio_ms": first_audio_ms},
        )

    async def _reply(self, text: str, history: list[dict[str, str]] | None = None) -> str:
        try:
            out = self.reply(text, history) if self._takes_history else self.reply(text)
        except TypeError:
            out = self.reply(text)
        if asyncio.iscoroutine(out):
            out = await out
        return str(out or "")

    async def health(self) -> tuple[bool, str]:
        try:
            asr, tts = self._engines()
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        return True, f"asr={type(asr).__name__} tts={type(tts).__name__}"
