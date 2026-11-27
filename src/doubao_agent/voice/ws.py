from __future__ import annotations

from typing import Any, Awaitable, Callable

from .session import VoiceSession


async def handle_voice_ws(
    recv: Callable[[], Awaitable[Any]],
    send_json: Callable[[dict[str, Any]], Awaitable],
    send_bytes: Callable[[bytes], Awaitable[Any]],
    session: VoiceSession,
) -> None:
    """双工会话的传输无关实现：收 PCM16 帧 → 出事件 JSON / 回复音频二进制。

    recv 返回 bytes（音频帧）、str（控制指令）或 None（连接关闭）。
    这样核心逻辑不绑 FastAPI，单测用假队列即可跑通。
    """
    await send_json({"type": "ready", "sample_rate": 16000, "format": "pcm16"})
    while True:
        msg = await recv()
        if msg is None:
            break
        if isinstance(msg, str):
            cmd = msg.strip()
            if cmd in ("end", "hangup"):
                await send_json({"type": "bye"})
                break
            if cmd.startswith("text:"):
                await _emit(session.stream_text(cmd[5:]), send_json, send_bytes)
            continue

        if isinstance(msg, (bytes, bytearray)):
            events = session.push_audio(bytes(msg))
            await _emit(events, send_json, send_bytes)
            for ev in events:
                if ev.kind == "speech_end":
                    await _emit(session.stream_audio(ev.audio), send_json, send_bytes)
    return


async def _emit(events, send_json, send_bytes) -> None:
    """events 既可以是事件列表，也可以是异步生成器：流式时边出边发。"""
    if hasattr(events, "__aiter__"):
        async for ev in events:
            await _emit_one(ev, send_json, send_bytes)
        return
    for ev in events:
        await _emit_one(ev, send_json, send_bytes)


async def _emit_one(ev, send_json, send_bytes) -> None:
    if ev.kind == "audio":
        await send_json(
            {
                "type": "speech",
                "text": ev.text,
                "seconds": ev.meta.get("seconds"),
                "rate": ev.meta.get("rate"),
            }
        )
        await send_bytes(ev.audio)
        return
    payload = {"type": ev.kind, **ev.to_dict()}
    if ev.kind == "speech_end":
        payload.pop("audio", None)
        payload["bytes"] = ev.meta.get("bytes")
    await send_json(payload)
