from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from doubao_agent.core.multimodal import (
    MAX_IMAGE_BYTES,
    extract_frames,
    extract_subtitle_stream,
    ingest,
    ocr,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
HAS_TESS = shutil.which("tesseract") is not None


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True, timeout=60)


@pytest.fixture()
def video(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("no ffmpeg")
    out = tmp_path / "demo.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "3", str(out))
    return out


@pytest.fixture()
def audio(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("no ffmpeg")
    out = tmp_path / "voice.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(out))
    return out


def test_video_frames_extracted(video, tmp_path):
    frames = extract_frames(video, tmp_path / "frames")
    assert 1 <= len(frames) <= 4
    assert all(f.stat().st_size < MAX_IMAGE_BYTES for f in frames)
    assert all(f.suffix == ".jpg" for f in frames)


def test_video_ingest_notes_and_images(video):
    bundle = ingest([str(video)])
    assert any("【视频】" in n for n in [bundle.notes])
    assert 1 <= len(bundle.images) <= 4
    assert "秒" in bundle.notes


def test_video_subtitle_preferred(video, tmp_path):
    srt = tmp_path / "sub.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n第一句台词\n\n2\n00:00:02,000 --> 00:00:03,000\n第二句台词\n",
        "utf-8",
    )
    muxed = tmp_path / "with_sub.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(srt), "-c", "copy", "-c:s", "srt", str(muxed)],
        capture_output=True, check=True, timeout=60,
    )
    text = extract_subtitle_stream(muxed)
    assert text and "第一句台词" in text
    bundle = ingest([str(muxed)])
    assert "字幕：" in bundle.notes


def test_audio_with_sidecar_transcript(audio, tmp_path):
    (tmp_path / "voice.txt").write_text("这是一段会议录音的转写", "utf-8")
    bundle = ingest([str(audio)])
    assert "转写稿：这是一段会议录音的转写" in bundle.notes
    assert not any("ASR" in w for w in bundle.warnings)


def test_audio_without_transcript_warns(audio):
    bundle = ingest([str(audio)])
    assert any("ASR" in w for w in bundle.warnings)
    assert bundle.images == []


def test_screen_without_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    bundle = ingest([], screen=True)
    assert bundle.images == []
    assert any("DISPLAY" in w for w in bundle.warnings)


def test_oversize_image_skipped(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"\0" * (MAX_IMAGE_BYTES + 1))
    bundle = ingest([str(big)])
    assert str(big) not in bundle.images
    assert any("超过 5MB" in w for w in bundle.warnings)


def test_unknown_suffix_warns(tmp_path):
    odd = tmp_path / "x.zzz"
    odd.write_text("hi", "utf-8")
    bundle = ingest([str(odd)])
    assert any("不支持" in w for w in bundle.warnings)


def test_ocr_reads_chinese(tmp_path):
    if not HAS_TESS:
        pytest.skip("no tesseract")
    if shutil.which("convert") is None:
        pytest.skip("no imagemagick")
    png = tmp_path / "shot.png"
    subprocess.run(
        ["convert", "-font", "Noto-Serif-CJK-SC", "-pointsize", "36", "label:多模态测试", str(png)],
        check=True, timeout=30,
    )
    text = ocr([png])
    assert "模态" in text or "测试" in text
