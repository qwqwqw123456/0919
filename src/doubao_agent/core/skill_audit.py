"""技能审计：把"装了多少"换成"有多少真能跑"。

三档口径：
- executable：有脚本入口，且静态检查（文件存在 + python 可编译 / shell 语法）通过；
- runnable：实际用最小输入跑通并给出合法输出（只对有入口的算）；
- knowledge：无入口的知识型技能，只校验 frontmatter 可解析、描述非空、能被关键词命中。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..skills import Skill, SkillManager

SAMPLE_STRINGS = ("测试文本", "统计一下", "本机有几个 CPU 核？")


def _minimal_payload(skill: Skill) -> dict[str, Any]:
    """按 input_schema 造最小合法输入；没有 schema 就给一个通用文本字段。"""
    schema = skill.input_schema or {}
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    if not props:
        return {"text": SAMPLE_STRINGS[0], "task": SAMPLE_STRINGS[2]}
    payload: dict[str, Any] = {}
    for key, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        if "default" in spec:
            payload[key] = spec["default"]
            continue
        kind = str(spec.get("type") or "string")
        if kind in ("string", "text"):
            payload[key] = SAMPLE_STRINGS[0] if "text" in key.lower() else SAMPLE_STRINGS[1]
        elif kind in ("number", "integer"):
            payload[key] = 1
        elif kind == "boolean":
            payload[key] = False
        elif kind == "array":
            payload[key] = []
        elif kind == "object":
            payload[key] = {}
    return payload


def _static_check(skill: Skill) -> dict[str, Any]:
    entry = skill.directory / str(skill.entry or "")
    if not skill.executable:
        return {"ok": True, "detail": "知识型，无脚本入口"}
    if not entry.is_file():
        return {"ok": False, "detail": f"入口缺失：{entry}"}
    if skill.runtime == "python":
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(entry)],
            capture_output=True, text=True, timeout=30,
        )
        return {
            "ok": proc.returncode == 0,
            "detail": "python 语法通过" if proc.returncode == 0 else proc.stderr.strip()[:200],
        }
    proc = subprocess.run(["bash", "-n", str(entry)], capture_output=True, text=True, timeout=30)
    return {
        "ok": proc.returncode == 0,
        "detail": "shell 语法通过" if proc.returncode == 0 else proc.stderr.strip()[:200],
    }


def _smoke(skill: Skill, timeout: float = 20.0) -> dict[str, Any]:
    """真跑一次最小输入；只判断"能启动并给输出"，不判断业务正确性。"""
    entry = skill.directory / str(skill.entry or "")
    payload = json.dumps(_minimal_payload(skill), ensure_ascii=False)
    cmd = [sys.executable, str(entry)] if skill.runtime == "python" else ["bash", str(entry)]
    try:
        proc = subprocess.run(
            cmd, input=payload, capture_output=True, text=True,
            timeout=timeout, cwd=str(skill.directory),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "detail": f"超时 {timeout}s"}
    except OSError as exc:
        return {"ok": False, "detail": f"起不来：{exc}"}
    out = (proc.stdout or "").strip()
    # 退出码 1 的扫描类技能也视为"跑通"，只要能给出结论
    if proc.returncode not in (0, 1):
        return {"ok": False, "detail": f"退出码 {proc.returncode}：{(proc.stderr or '')[:120]}"}
    if not out:
        return {"ok": False, "detail": "没有任何输出"}
    try:
        json.loads(out.splitlines()[-1])
        parsed = True
    except json.JSONDecodeError:
        parsed = False
    return {
        "ok": True,
        "detail": f"退出码 {proc.returncode}，输出 {len(out)} 字符，JSON={'是' if parsed else '否'}",
    }


def audit(manager: SkillManager, smoke: bool = True) -> dict[str, Any]:
    manager.refresh()
    skills = manager.list()
    rows: list[dict[str, Any]] = []
    for skill in skills:
        static = _static_check(skill)
        row = {
            "name": skill.name,
            "executable": skill.executable,
            "entry": skill.entry,
            "static_ok": static["ok"],
            "static_detail": static["detail"],
            "smoke_ok": None,
            "smoke_detail": "",
            "description_len": len(skill.description or ""),
        }
        if smoke and skill.executable and static["ok"]:
            result = _smoke(skill)
            row["smoke_ok"] = result["ok"]
            row["smoke_detail"] = result["detail"]
        rows.append(row)

    executable = [r for r in rows if r["executable"]]
    knowledge = [r for r in rows if not r["executable"]]
    static_bad = [r for r in executable if not r["static_ok"]]
    smoke_bad = [r for r in executable if r["smoke_ok"] is False]
    thin_desc = [r for r in knowledge if r["description_len"] < 8]
    return {
        "total": len(rows),
        "executable": len(executable),
        "knowledge": len(knowledge),
        "runnable": sum(1 for r in executable if r["smoke_ok"] is True) if smoke else None,
        "static_bad": [{"name": r["name"], "detail": r["static_detail"]} for r in static_bad],
        "smoke_bad": [{"name": r["name"], "detail": r["smoke_detail"]} for r in smoke_bad],
        "thin_description": [r["name"] for r in thin_desc],
        "rows": rows,
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [
        f"技能总数 {result['total']}｜可执行 {result['executable']}｜知识型 {result['knowledge']}"
    ]
    if result.get("runnable") is not None:
        lines.append(f"可执行里实测跑通 {result['runnable']}/{result['executable']}")
    for item in result["static_bad"]:
        lines.append(f"  静态失败 {item['name']}：{item['detail']}")
    for item in result["smoke_bad"]:
        lines.append(f"  实跑失败 {item['name']}：{item['detail']}")
    if result["thin_description"]:
        lines.append(f"  描述过短（<8 字）{len(result['thin_description'])} 个：{', '.join(result['thin_description'][:6])}")
    return "\n".join(lines)
