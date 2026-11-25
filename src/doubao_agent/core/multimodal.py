"""多模态附件入口：把图片/视频/音频/屏幕统一收成「图片列表 + 文字注记」。

本机能力边界（实测）：
- 图片：直接进多模态消息（size 上限与 provider 一致）
- 视频：ffprobe 读元信息 + ffmpeg 抽关键帧；有字幕流就把字幕抽成文字
- 音频：没有本地 ASR，只读元信息；同目录有 .txt/.srt/.vtt 转写稿才进上下文
- 屏幕：X11 下用 ImageMagick `import` 截屏；无图形会话时明确提示，不伪造
- OCR：tesseract（chi_sim+eng）做文字型截图/帧的补充文本
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
TEXT_SIDECARS = (".txt", ".srt", ".vtt", ".md")

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_FRAMES = 4
MAX_OCR_CHARS = 1500


@dataclass
class Bundle:
    images: list[str] = field(default_factory=list)
    notes: str = ""
    warnings: list[str] = field(default_factory=list)


def _run(cmd: list[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _probe(path: Path) -> dict:
    out = _run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        timeout=30,
    )
    if out.returncode != 0:
        return {}
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def _fmt_seconds(value: float | None) -> str:
    if not value:
        return "未知时长"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}分{seconds}秒" if minutes else f"{seconds}秒"


def _srt_to_text(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.isdigit() or " --> " in line or line.startswith(("WEBVTT", "NOTE")):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    return " ".join(dict.fromkeys(lines)).strip()


def _read_sidecar_transcript(path: Path) -> str | None:
    for suffix in TEXT_SIDECARS:
        for candidate in (path.with_suffix(suffix), path.parent / f"{path.stem}{suffix}"):
            if candidate.is_file() and candidate != path:
                try:
                    raw = candidate.read_text("utf-8")
                except OSError:
                    continue
                return _srt_to_text(raw) if candidate.suffix in (".srt", ".vtt") else raw.strip()
    return None


def extract_frames(video: Path, out_dir: Path, count: int = MAX_FRAMES) -> list[Path]:
    """按均匀间隔抽帧，缩到宽 ≤1280；返回生成的 jpg 路径。"""
    if shutil.which("ffmpeg") is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    info = _probe(video)
    duration = 0.0
    try:
        duration = float(info.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        fps_filter = f"fps=1/{max(1, 60 // count)}"
    else:
        step = max(duration / count, 0.5)
        fps_filter = f"fps=1/{step:.3f}"
    pattern = out_dir / f"{video.stem}-frame-%02d.jpg"
    out = _run(
        [
            "ffmpeg", "-y", "-i", str(video),
            "-vf", f"{fps_filter},scale='min(1280,iw)':-2",
            "-frames:v", str(count), "-q:v", "3", str(pattern),
        ],
        timeout=120,
    )
    if out.returncode != 0:
        return []
    return sorted(out_dir.glob(f"{video.stem}-frame-*.jpg"))


def extract_subtitle_stream(video: Path) -> str | None:
    info = _probe(video)
    if not any(s.get("codec_type") == "subtitle" for s in info.get("streams", [])):
        return None
    out = _run(
        ["ffmpeg", "-y", "-i", str(video), "-map", "0:s:0", "-f", "srt", "pipe:1"],
        timeout=120,
    )
    text = _srt_to_text(out.stdout or "") if out.returncode == 0 else ""
    return text or None


def ocr(paths: Iterable[Path], limit: int = MAX_OCR_CHARS) -> str:
    """tesseract 中文+英文；不可用或没识别到文字时返回空串。"""
    if shutil.which("tesseract") is None:
        return ""
    chunks: list[str] = []
    for path in paths:
        out = _run(["tesseract", str(path), "stdout", "-l", "chi_sim+eng"], timeout=60)
        text = (out.stdout or "").strip()
        if out.returncode == 0 and text:
            chunks.append(f"[{path.name}]\n{text}")
    merged = "\n".join(chunks).strip()
    if len(merged) > limit:
        merged = merged[:limit] + f"\n...[OCR 截断，共 {len(merged)} 字]"
    return merged


def capture_screen(out_dir: Path) -> tuple[Path | None, str | None]:
    """X11 截屏：无 DISPLAY 或无 import 命令时返回原因，不生成假图。"""
    import os

    if not os.environ.get("DISPLAY"):
        return None, "当前会话没有 DISPLAY（无图形桌面），截屏不可用；可用宿主截图工具或上传图片"
    if shutil.which("import") is None:
        return None, "本机没有 ImageMagick import，无法截屏"
    target = out_dir / "screen.png"
    out = _run(["import", "-window", "root", str(target)], timeout=30)
    if out.returncode != 0 or not target.is_file():
        return None, f"截屏失败：{(out.stderr or '').strip()[:120] or '未知原因'}"
    return target, None


def ingest(
    paths: list[str] | None = None,
    *,
    screen: bool = False,
    ocr_enabled: bool = False,
) -> Bundle:
    """把一组附件收成 Bundle；识别不了的明确进 warnings，不编造内容。"""
    bundle = Bundle()
    work = Path(tempfile.mkdtemp(prefix="agent-attach-"))
    media_for_ocr: list[Path] = []
    notes: list[str] = []

    if screen:
        shot, reason = capture_screen(work)
        if shot is not None:
            bundle.images.append(str(shot))
            notes.append("【屏幕】已截取当前桌面")
            media_for_ocr.append(shot)
        else:
            bundle.warnings.append(f"【屏幕】{reason}")

    for raw in paths or []:
        path = Path(raw).expanduser()
        if not path.is_file():
            bundle.warnings.append(f"【附件】找不到文件：{raw}")
            continue
        suffix = path.suffix.lower()

        if suffix in IMAGE_SUFFIXES:
            size = path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                bundle.warnings.append(
                    f"【图片】{path.name} 超过 5MB（{size // 1024} KB），未附上；请先压缩"
                )
                continue
            bundle.images.append(str(path))
            media_for_ocr.append(path)

        elif suffix in VIDEO_SUFFIXES:
            frames = extract_frames(path, work)
            info = _probe(path)
            duration = None
            try:
                duration = float(info.get("format", {}).get("duration") or 0) or None
            except (TypeError, ValueError):
                duration = None
            line = f"【视频】{path.name}（{_fmt_seconds(duration)}）"
            subtitle = extract_subtitle_stream(path)
            if subtitle:
                line += f"；字幕：{subtitle[:1200]}"
            elif frames:
                line += f"；已抽 {len(frames)} 帧作为画面"
                media_for_ocr.extend(frames)
            else:
                line += "；抽帧失败，未附画面"
                bundle.warnings.append(f"【视频】{path.name} 抽帧失败（ffmpeg 不可用或文件损坏）")
            bundle.images.extend(str(f) for f in frames)
            notes.append(line)

        elif suffix in AUDIO_SUFFIXES:
            info = _probe(path)
            duration = None
            try:
                duration = float(info.get("format", {}).get("duration") or 0) or None
            except (TypeError, ValueError):
                duration = None
            line = f"【音频】{path.name}（{_fmt_seconds(duration)}）"
            transcript = _read_sidecar_transcript(path)
            if transcript:
                line += f"；转写稿：{transcript[:1200]}"
            else:
                line += "；本机无 ASR 引擎，音频内容未转写"
                bundle.warnings.append(
                    f"【音频】{path.name} 没找到同名 .txt/.srt/.vtt 转写稿，"
                    "本机也没有 ASR（语音识别）引擎，这段声音不会被理解"
                )
            notes.append(line)
        else:
            bundle.warnings.append(f"【附件】不支持的格式：{path.name}")

    if ocr_enabled and media_for_ocr:
        text = ocr(media_for_ocr)
        if text:
            notes.append(f"【OCR 文字】\n{text}")
        else:
            bundle.warnings.append("【OCR】没有识别到文字（或 tesseract 不可用）")

    bundle.notes = "\n".join(notes)
    return bundle
