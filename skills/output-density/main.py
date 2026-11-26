#!/usr/bin/env python3
"""output-density —— 交付物密度检测（迁自 v1.62 check-output-density.py）。

stdin: {"path": "...", "text": "...", "thresholds": {...}}
stdout: {"ok": bool, "counts": {...}, "warnings": [...], "source": "path|text"}
"""
import json
import re
import sys
from pathlib import Path

DEFAULTS = {"placeholder_max": 3, "min_digits": 5, "min_links": 2, "min_bullets": 3}

PLACEHOLDER_PAT = re.compile(r"\b(TODO|TBD|XXX|待定|待补充|待完善|FIXME|占位)\b", re.I)
DIGIT_PAT = re.compile(r"\d+")
URL_PAT = re.compile(r"https?://\S+")
BULLET_PAT = re.compile(r"^\s*[-*]\s+", re.M)


def main() -> int:
    try:
        raw = sys.stdin.read().strip()
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"入参不是合法 JSON: {exc}"}, ensure_ascii=False))
        return 2
    if not isinstance(args, dict):
        print(json.dumps({"ok": False, "error": "入参必须是 JSON 对象"}, ensure_ascii=False))
        return 2

    th = dict(DEFAULTS)
    for key, value in (args.get("thresholds") or {}).items():
        if key in th and isinstance(value, int) and value >= 0:
            th[key] = value

    text = args.get("text")
    path = args.get("path")
    source = "text"
    if text is None:
        if not path:
            print(json.dumps({"ok": False, "error": "需要 path 或 text 之一"}, ensure_ascii=False))
            return 2
        target = Path(str(path)).expanduser()
        if not target.is_file():
            print(json.dumps({"ok": False, "error": f"文件不存在: {target}"}, ensure_ascii=False))
            return 1
        text = target.read_text(encoding="utf-8", errors="ignore")
        source = "path"
    text = str(text)

    placeholders = PLACEHOLDER_PAT.findall(text)
    digits = DIGIT_PAT.findall(text)
    urls = URL_PAT.findall(text)
    bullets = BULLET_PAT.findall(text)

    warnings = []
    if len(placeholders) > th["placeholder_max"]:
        warnings.append(f"占位符过多({len(placeholders)}个，上限{th['placeholder_max']})——像骨架交付，不是成品")
    if len(digits) < th["min_digits"]:
        warnings.append(f"具体数字太少({len(digits)}个，下限{th['min_digits']})——可能是空洞描述")
    if len(text) > 500 and len(urls) < th["min_links"]:
        warnings.append(f"链接太少({len(urls)}个，下限{th['min_links']})——长文没有来源支撑")
    if len(text) > 300 and len(bullets) < th["min_bullets"]:
        warnings.append(f"bullet太少({len(bullets)}个，下限{th['min_bullets']})——长文没有结构化")

    print(
        json.dumps(
            {
                "ok": not warnings,
                "source": source,
                "chars": len(text),
                "counts": {
                    "placeholders": len(placeholders),
                    "digits": len(digits),
                    "urls": len(urls),
                    "bullets": len(bullets),
                },
                "thresholds": th,
                "warnings": warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
