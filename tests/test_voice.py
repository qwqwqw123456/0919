from __future__ import annotations

import asyncio
import math
import wave
from array import array
from pathlib import Path

import pytest

from doubao_agent.sources.voice import VoiceSource
from doubao_agent.types import ToolCall
from doubao_agent.voice.engines import MockASR, MockTTS, pcm16_from_wav, wav_bytes
from doubao_agent.voice.session import VoiceSession, split_sentences
from doubao_agent.voice.vad import EnergyVAD
from doubao_agent.voice.ws import handle_voice_ws

SR = 16000


def tone(ms: int, amp: float = 0.35, freq: float = 300.0) -> bytes:
    n = int(SR * ms / 1000)
    samples = [int(32767 * amp * math.sin(2 * math.pi * freq * i / SR)) for i in range(n)]
    return array("h", samples).tobytes()


def silence(ms: int) -> bytes:
    return b"\x00\x00" * int(SR * ms / 1000)


def test_vad_emits_start_then_end_with_utterance() -> None:
    vad = EnergyVAD(min_speech_ms=200, hangover_ms=400)
    ev1 = vad.feed(silence(500) + tone(400))
    kinds = [e.kind for e in ev1]
    assert "speech_start" in kinds
    ev2 = vad.feed(silence(600))
    assert [e.kind for e in ev2] == ["speech_end"]
    assert len(ev2[0].audio) > 0


def test_vad_ignores_short_blip() -> None:
    vad = EnergyVAD(min_speech_ms=200, hangover_ms=300)
    events = vad.feed(silence(400) + tone(80) + silence(400))
    assert events == []


def test_vad_flush_closes_open_utterance() -> None:
    vad = EnergyVAD()
    vad.feed(silence(600) + tone(500))
    closed = vad.flush()
    assert closed and closed[0].kind == "speech_end" and closed[0].audio


def test_split_sentences_merges_fragments() -> None:
    assert split_sentences("好。我盯着。降价了就通知你！") == ["好我盯着", "降价了就通知你"]
    assert split_sentences("好的小周。我一有降价就告诉你。") == ["好的小周", "我一有降价就告诉你"]


def _session(reply_text: str = "好的小周。我一有降价就告诉你。") -> VoiceSession:
    async def reply(text, history):
        return reply_text

    return VoiceSession(MockASR(), MockTTS(), reply, quick_ack="")


def test_session_answers_with_sentence_audio() -> None:
    s = _session()
    events = asyncio.run(s.respond(b"\x00\x00" * 100))
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["user_text", "reply_text"]
    audios = [e for e in events if e.kind == "audio"]
    assert len(audios) == 2
    assert all(e.audio for e in audios)
    assert s.history[-1]["role"] == "assistant"


def test_session_barge_in_drops_queued_audio() -> None:
    s = _session()
    asyncio.run(s.respond(b"\x00\x00" * 100))
    assert s.playing
    events = s.push_audio(silence(600) + tone(500))
    assert any(e.kind == "barge_in" for e in events)
    assert s.drain_audio() == []


def test_edge_tts_falls_back_to_piper(monkeypatch) -> None:
    from doubao_agent.voice.engines import EdgeTTS

    edge = EdgeTTS("zh-CN-XiaoxiaoNeural", fallback=MockTTS())
    monkeypatch.setattr("edge_tts.Communicate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: (_ for _ in ()).throw(OSError("no cli")))
    pcm, rate = edge.synthesize("兜底测试")
    assert pcm and rate == SR


def test_ack_cache_survives_new_session() -> None:
    s1 = VoiceSession(MockASR(), MockTTS(), lambda t, h: asyncio.sleep(0, "好"), quick_ack="嗯，我在听。")
    s1.prewarm()
    payload = s1.ack_payload()
    assert payload
    s2 = VoiceSession(MockASR(), MockTTS(), lambda t, h: asyncio.sleep(0, "好"), quick_ack="嗯，我在听。", ack_audio=payload)
    assert s2.ack_payload() == payload


def test_session_ack_is_prewarmed() -> None:
    s = VoiceSession(MockASR(), MockTTS(), lambda t, h: asyncio.sleep(0, "好的"), quick_ack="嗯，我在听。")
    s.prewarm()
    assert s._ack_pcm
    events = asyncio.run(s.respond_to_text("在吗"))
    assert events[1].kind == "ack"


def test_asr_failure_does_not_kill_session() -> None:
    class Boom(MockASR):
        def transcribe(self, pcm, sr=SR):
            raise RuntimeError("engine down")

    async def reply(text, history):
        return "x"

    s = VoiceSession(Boom(), MockTTS(), reply, quick_ack="")
    events = asyncio.run(s.respond(b"\x00\x00" * 10))
    assert [e.kind for e in events] == ["error"]
    assert "engine down" in events[0].text


def test_ws_duplex_round_trip() -> None:
    s = _session("我在，票已经帮你盯上了。")
    sent_json: list[dict] = []
    sent_bytes: list[bytes] = []
    script: list = [silence(600) + tone(500) + silence(700), "hangup"]

    async def recv():
        return script.pop(0) if script else None

    async def send_json(payload):
        sent_json.append(payload)

    async def send_bytes(data):
        sent_bytes.append(data)

    asyncio.run(handle_voice_ws(recv, send_json, send_bytes, s))
    assert sent_json[0]["type"] == "ready"
    kinds = {p["type"] for p in sent_json}
    assert {"speech_start", "speech_end", "user_text", "reply_text", "speech", "bye"} <= kinds
    assert sent_bytes and sent_json[-1]["type"] == "bye"


def test_ws_text_command_bypasses_asr() -> None:
    s = _session("收到。")
    sent: list[dict] = []
    script = ["text:帮我看下机票", "hangup"]

    async def recv():
        return script.pop(0) if script else None

    async def send_json(payload):
        sent.append(payload)

    async def send_bytes(data): ...

    asyncio.run(handle_voice_ws(recv, send_json, send_bytes, s))
    assert any(p["type"] == "user_text" and p.get("text") == "帮我看下机票" for p in sent)
    assert "speech" in {p["type"] for p in sent} and sent[-1]["type"] == "bye"


def test_voice_source_call_with_text(tmp_path: Path) -> None:
    async def reply(text, history=None):
        return "记下了，我盯着价。"

    src = VoiceSource(reply, {"asr": "mock", "tts": "mock"}, out_dir=tmp_path)
    res = asyncio.run(src.call_tool(ToolCall(call_id="c", name="call", arguments={"text": "帮我盯机票"})))
    assert res.ok and "记下了" in res.meta["reply"]
    assert Path(res.meta["audio"]).is_file()
    assert res.meta["timeline"][-1]["kind"] == "done"


def test_voice_source_say_then_transcribe(tmp_path: Path) -> None:
    src = VoiceSource(lambda text, history=None: "ok", {"asr": "mock", "tts": "mock"}, out_dir=tmp_path)
    say = asyncio.run(src.call_tool(ToolCall(call_id="c", name="say", arguments={"text": "测试一句话"})))
    assert say.ok and Path(say.meta["path"]).is_file()
    tr = asyncio.run(src.call_tool(ToolCall(call_id="c", name="transcribe", arguments={"path": say.meta["path"]})))
    assert tr.ok and tr.content


def test_voice_source_say_honors_absolute_path(tmp_path: Path) -> None:
    src = VoiceSource(lambda text, history=None: "ok", {"asr": "mock", "tts": "mock"}, out_dir=tmp_path)
    target = tmp_path / "sub" / "given-name"
    res = asyncio.run(src.call_tool(ToolCall(call_id="c", name="say", arguments={"text": "hi", "filename": str(target)})))
    assert res.ok and Path(res.meta["path"]) == target.with_suffix(".wav")
    assert Path(res.meta["path"]).is_file()


def test_voice_source_rejects_missing_file(tmp_path: Path) -> None:
    src = VoiceSource(lambda text, history=None: "ok", {}, out_dir=tmp_path)
    res = asyncio.run(src.call_tool(ToolCall(call_id="c", name="transcribe", arguments={"path": str(tmp_path / "nope.wav")})))
    assert not res.ok and "不存在" in res.error


def test_wav_write_read_roundtrip(tmp_path: Path) -> None:
    pcm = tone(300)
    p = tmp_path / "a.wav"
    p.write_bytes(wav_bytes(pcm, SR))
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
    assert len(pcm16_from_wav(p)) > 0
