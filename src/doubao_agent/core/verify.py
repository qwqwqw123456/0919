"""产物核验：不看模型怎么说，只看磁盘和退出码。

口径：核验器只认确定性证据——文件在不在、有多大、内容对不对、命令退出码是多少。
模型的自评最多算补充说明，不能推翻确定性结论。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

# 执行器文字里出现的绝对路径，常见交付物扩展名
_PATH_RE = re.compile(r"(/[^\s\"'`，。；：、()\[\]{}<>]+?\.(?:md|txt|json|csv|py|sh|html|yaml|yml|log|png|jpg|pdf|docx|xlsx))")
_WRITE_HINTS = ("写入", "写文件", "已保存", "保存到", "生成", "导出", "落盘", "created", "wrote")


def _fail(name: str, detail: str, hard: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": False, "detail": detail, "hard": hard}


def _ok(name: str, detail: str, hard: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": True, "detail": detail, "hard": hard}


def check_file(
    path: str | Path,
    min_bytes: int = 1,
    contains: Iterable[str] = (),
    must_parse_json: bool = False,
    hard: bool = True,
) -> dict[str, Any]:
    """文件必须存在、非空；可选：包含指定片段、能解析成 JSON。"""
    target = Path(str(path)).expanduser()
    name = f"file:{target.name}"
    hard_flag = {"hard": hard}
    if not target.is_file():
        return _fail(name, f"文件不存在：{target}", hard=hard)
    size = target.stat().st_size
    if size < min_bytes:
        return _fail(name, f"文件过小：{size} 字节 < {min_bytes}", hard=hard)
    try:
        text = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        return _fail(name, f"读不了文件：{exc}", hard=hard)
    for token in contains:
        if token not in text:
            return _fail(name, f"缺少片段：{token!r}", hard=hard)
    if must_parse_json:
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return _fail(name, f"不是合法 JSON：{exc}", hard=hard)
    return _ok(name, f"{size} 字节，片段与格式检查通过", hard=hard)


def check_text(
    value: str,
    contains: Iterable[str] = (),
    not_contains: Iterable[str] = (),
    regex: str | None = None,
) -> dict[str, Any]:
    text = value or ""
    for token in contains:
        if token not in text:
            return _fail("text", f"缺少片段：{token!r}")
    for token in not_contains:
        if token in text:
            return _fail("text", f"不该出现：{token!r}")
    if regex and not re.search(regex, text):
        return _fail("text", f"正则未命中：{regex}")
    return _ok("text", "文本断言通过")


def check_exit(code: int | None, expect: int = 0) -> dict[str, Any]:
    if code is None:
        return _fail("exit", "没有拿到退出码")
    if int(code) != expect:
        return _fail("exit", f"退出码 {code}，期望 {expect}")
    return _ok("exit", f"退出码 {code}")


def paths_in(text: str) -> list[str]:
    """从执行器文字里捞绝对路径，用于回查磁盘实物。"""
    seen: list[str] = []
    for match in _PATH_RE.findall(text or ""):
        # 通配符、占位符、变量拼出来的路径都不是实物，不做磁盘判定
        if any(ch in match for ch in "*?{}<>$%~"):
            continue
        if match not in seen:
            seen.append(match)
    return seen


def claims_write(text: str) -> bool:
    lowered = (text or "").lower()
    return any(hint in lowered for hint in _WRITE_HINTS)


def check_claimed_paths(outputs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """执行器说写了文件，就去看文件在不在；没提路径的不猜。"""
    checks: list[dict[str, Any]] = []
    for item in outputs:
        output = str(item.get("output") or "")
        for path in paths_in(output):
            if claims_write(output):
                checks.append(check_file(path))
    return checks


def run_specs(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """按规格跑确定性检查；规格来自评测用例或流水线拆解。"""
    checks: list[dict[str, Any]] = []
    for spec in specs:
        if isinstance(spec, str):
            checks.append(check_file(spec))
            continue
        kind = str(spec.get("kind") or "file")
        if kind == "file":
            checks.append(
                check_file(
                    spec.get("path", ""),
                    min_bytes=int(spec.get("min_bytes", 1)),
                    contains=spec.get("contains") or (),
                    must_parse_json=bool(spec.get("json", False)),
                )
            )
        elif kind == "text":
            checks.append(
                check_text(
                    str(spec.get("value") or ""),
                    contains=spec.get("contains") or (),
                    not_contains=spec.get("not_contains") or (),
                    regex=spec.get("regex"),
                )
            )
        elif kind == "exit":
            checks.append(check_exit(spec.get("code"), int(spec.get("expect", 0))))
        else:
            checks.append(_fail("spec", f"未知检查类型：{kind}"))
    return checks


def summarize(checks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """只有硬检查参与判定；软检查（拆解器声明的路径）只作证据，不自动判死。"""
    items = list(checks)
    hard_issues = [f"{c['name']}: {c['detail']}" for c in items if not c["ok"] and c.get("hard", True)]
    soft_issues = [f"{c['name']}（软）: {c['detail']}" for c in items if not c["ok"] and not c.get("hard", True)]
    return {"pass": not hard_issues, "checks": items, "issues": hard_issues, "soft_issues": soft_issues}


def format_evidence(checks: Iterable[dict[str, Any]]) -> str:
    items = list(checks)
    if not items:
        return "-（无确定性检查）"
    return "\n".join(
        f"- {'通过' if c['ok'] else ('未通过' if c.get('hard', True) else '提示')} {c['name']}: {c['detail']}"
        for c in items
    )
