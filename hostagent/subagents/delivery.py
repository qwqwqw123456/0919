"""delivery_agent 执行层（manifest 2.2，6 个动作）。

本地产物一律落 output/ 沙箱；对外动作走 lark-cli，dry-run 不真实外发。
所有失败如实返回 ok=False，不伪造回执。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

LarkRunner = Callable[[list[str]], tuple[int, str, str]]

_SAFE_NAME = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")
_TZ = timezone(timedelta(hours=8))


def default_lark_runner(args: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["lark-cli", *args], capture_output=True, text=True, timeout=60, check=False
        )
    except FileNotFoundError:
        return 127, "", "lark-cli 未安装"
    except subprocess.TimeoutExpired:
        return 124, "", "lark-cli 调用超时"
    return proc.returncode, proc.stdout, proc.stderr


def _ok(output: Any) -> dict[str, Any]:
    return {"ok": True, "output": output}


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _safe_name(name: str, default: str, suffix: str) -> str:
    base = _SAFE_NAME.sub("_", (name or default).strip()) or default
    base = base[:80]
    if not base.lower().endswith(suffix):
        base += suffix
    return base


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _normalize_time(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        raise ValueError("时间为空")
    text = text.replace("/", "-").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_TZ)
    return dt


def do_run(
    action: str,
    ctx: dict[str, Any],
    *,
    output_dir: str | Path = "output",
    dry_run: bool = False,
    lark: LarkRunner | None = None,
) -> dict[str, Any]:
    lark = lark or default_lark_runner
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        if action == "create_sheet":
            return _create_sheet(ctx, out)
        if action == "create_ppt":
            return _create_ppt(ctx, out)
        if action == "create_doc":
            return _create_doc(ctx, dry_run, lark)
        if action == "send_message":
            return _send_message(ctx, dry_run, lark)
        if action == "create_calendar":
            return _create_calendar(ctx, dry_run, lark)
        if action == "send_file":
            return _send_file(ctx, out, dry_run, lark)
    except Exception as exc:
        return _err(f"{action} 执行异常: {type(exc).__name__}: {exc}")
    return {"ok": False, "error": f"未知动作 {action}", "agent": "delivery_agent"}


def _content_to_rows(content: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in re.split(r"[,\t，]", line)])
    return rows


def _create_sheet(ctx: dict[str, Any], out: Path) -> dict[str, Any]:
    from openpyxl import Workbook

    title = str(ctx.get("title") or "sheet")
    header = ctx.get("header") or []
    rows = ctx.get("rows") or []
    if not rows and ctx.get("content"):
        rows = _content_to_rows(str(ctx["content"]))
    if not header and not rows:
        return _err("create_sheet 缺少 header/rows/content")

    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    if header:
        sheet.append([str(h) for h in header])
    for row in rows:
        if isinstance(row, (list, tuple)):
            sheet.append(["" if cell is None else str(cell) for cell in row])
        else:
            sheet.append([str(row)])
    path = out / _safe_name(str(ctx.get("filename") or title), "sheet", ".xlsx")
    book.save(path)
    return _ok(str(path))


def _create_ppt(ctx: dict[str, Any], out: Path) -> dict[str, Any]:
    from pptx import Presentation

    title = str(ctx.get("title") or "presentation")
    slides = ctx.get("slides") or []
    deck = Presentation()
    cover = deck.slides.add_slide(deck.slide_layouts[0])
    cover.shapes.title.text = title
    if len(cover.placeholders) > 1:
        cover.placeholders[1].text = ""
    if not slides and ctx.get("summary"):
        slides = [{"heading": title, "bullets": [str(ctx["summary"])[:300]]}]
    if not slides:
        return _err("create_ppt 缺少 slides")
    for item in slides:
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = str(item.get("heading") or title)
        body = slide.placeholders[1].text_frame
        bullets = item.get("bullets") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        for index, bullet in enumerate(bullets):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            paragraph.text = str(bullet)
            paragraph.level = 0
    path = out / _safe_name(str(ctx.get("filename") or title), "deck", ".pptx")
    deck.save(path)
    return _ok(str(path))


def _parse_lark_json(stdout: str) -> dict[str, Any]:
    stdout = stdout.strip()
    if not stdout:
        return {}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw": stdout[:500]}
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data if isinstance(data, dict) else {"raw": stdout[:500]}


def _create_doc(ctx: dict[str, Any], dry_run: bool, lark: LarkRunner) -> dict[str, Any]:
    title = str(ctx.get("title") or "").strip()
    content = str(ctx.get("content") or "").strip()
    if not title and not content:
        return _err("create_doc 的 title 与 content 皆空，拒发")
    if dry_run:
        return _ok({"dry_run": True, "title": title, "chars": len(content)})
    code, stdout, stderr = lark(["docs", "+create", "--title", title, "--content", content])
    if code != 0:
        return _err(f"lark-cli docs 失败({code}): {(stderr or stdout)[:300]}")
    return _ok(_parse_lark_json(stdout))


def _target_chat(ctx: dict[str, Any]) -> str:
    return str(ctx.get("target_id") or ctx.get("chat_id") or os.environ.get("LARK_HITL_CHAT_ID") or "")


def _send_message(ctx: dict[str, Any], dry_run: bool, lark: LarkRunner) -> dict[str, Any]:
    text = str(ctx.get("text") or "").strip()
    if not text:
        return _err("send_message 文本为空，拒发")
    chat_id = _target_chat(ctx)
    if not chat_id:
        return _err("send_message 缺少目标渠道（target_id / LARK_HITL_CHAT_ID）")
    if dry_run:
        return _ok({"dry_run": True, "chat_id": chat_id, "preview": text[:80]})
    code, stdout, stderr = lark(["im", "+send", "--chat-id", chat_id, "--text", text])
    if code != 0:
        return _err(f"lark-cli im 失败({code}): {(stderr or stdout)[:300]}")
    return _ok(_parse_lark_json(stdout))


def _create_calendar(ctx: dict[str, Any], dry_run: bool, lark: LarkRunner) -> dict[str, Any]:
    title = str(ctx.get("title") or "").strip()
    if not title:
        return _err("create_calendar 缺少 title")
    start = _normalize_time(str(ctx.get("meeting_time") or ""))
    end_value = str(ctx.get("meeting_end") or "").strip()
    end = _normalize_time(end_value) if end_value else start + timedelta(minutes=30)
    if end <= start:
        return _err("create_calendar 结束时间必须晚于开始时间")
    payload = {
        "title": title,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "description": str(ctx.get("description") or ""),
    }
    if dry_run:
        return _ok({"dry_run": True, **payload})
    code, stdout, stderr = lark([
        "calendar", "+create",
        "--title", payload["title"],
        "--start", payload["start"],
        "--end", payload["end"],
        "--description", payload["description"],
    ])
    if code != 0:
        return _err(f"lark-cli calendar 失败({code}): {(stderr or stdout)[:300]}")
    return _ok(_parse_lark_json(stdout))


def _send_file(ctx: dict[str, Any], out: Path, dry_run: bool, lark: LarkRunner) -> dict[str, Any]:
    raw_path = str(ctx.get("file_path") or "").strip()
    if not raw_path:
        return _err("send_file 缺少 file_path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = out / path
    if not _inside(path, out):
        return _err(f"send_file 拒绝：产物必须在 output/ 沙箱内，收到 {raw_path}")
    if not path.exists():
        return _err(f"send_file 文件不存在: {path}")
    target = str(ctx.get("target_user_id") or "").strip()
    if not target:
        return _err("send_file 缺少 target_user_id")
    if dry_run:
        return _ok({"dry_run": True, "file": str(path), "target_user_id": target})
    code, stdout, stderr = lark(["im", "+send-file", "--user-id", target, "--file", str(path)])
    if code != 0:
        return _err(f"lark-cli send-file 失败({code}): {(stderr or stdout)[:300]}")
    return _ok(_parse_lark_json(stdout))
