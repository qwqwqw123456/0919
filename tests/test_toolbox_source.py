from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from doubao_agent.sources.toolbox import ToolboxSource
from doubao_agent.types import Risk, ToolCall, ToolResult


def _catalog(tmp_path: Path, tools: list[dict]) -> Path:
    path = tmp_path / "toolbox.yaml"
    path.write_text(yaml.safe_dump({"tools": tools}, allow_unicode=True), encoding="utf-8")
    return path


def _call(source: ToolboxSource, action: str, **args) -> ToolResult:
    return asyncio.run(source.call_tool(ToolCall(call_id="c", name=action, arguments=args)))


def test_list_filters_and_shows_substitute(tmp_path: Path) -> None:
    catalog = _catalog(
        tmp_path,
        [
            {"name": "true", "category": "通用", "entry": "true", "desc": "空命令", "risk": "L0", "status": "ok"},
            {"name": "x64dbg", "category": "Windows", "entry": None, "desc": "调试器", "risk": "L1",
             "status": "unavailable", "substitute": "rizin"},
        ],
    )
    src = ToolboxSource(catalog)
    res = _call(src, "list", category="Windows")
    assert res.ok and "x64dbg" in res.content and "替代：rizin" in res.content
    assert "可直接执行 0 项" in res.content

    only_ready = _call(src, "list", only_ready=True)
    assert "true" in only_ready.content and "x64dbg" not in only_ready.content


def test_run_whitelists_and_rejects_unregistered(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "echoer", "category": "通用", "entry": "echo", "desc": "回声", "risk": "L0", "status": "ok"},
    ]))
    ok = _call(src, "run", name="echoer", args=["hello", "toolbox"])
    assert ok.ok and "hello toolbox" in ok.content
    assert ok.meta["exit"] == 0

    denied = _call(src, "run", name="rm", args=["-rf", "/tmp/x"])
    assert not denied.ok and "没有" in denied.error


def test_run_refuses_unavailable_with_hint(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "pe-sieve", "category": "恶意样本", "entry": None, "desc": "注入检测", "risk": "L0",
         "status": "unavailable", "substitute": "mem-scan"},
    ]))
    res = _call(src, "run", name="pe-sieve")
    assert not res.ok and "mem-scan" in res.error


def test_args_must_be_list(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "echoer", "category": "通用", "entry": "echo", "desc": "回声", "risk": "L0", "status": "ok"},
    ]))
    res = _call(src, "run", name="echoer", args="hello; whoami")
    assert not res.ok and "列表" in res.error


def test_probe_reports_version_output(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "echoer", "category": "通用", "entry": "echo", "desc": "回声", "risk": "L0", "status": "ok"},
    ]))
    res = _call(src, "probe", name="echoer", args=["-n", "v1"])
    assert res.ok and res.content.strip() == "v1"


def test_timeout_kills_process(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "sleeper", "category": "通用", "entry": "sleep", "desc": "睡", "risk": "L0", "status": "ok"},
    ]))
    res = _call(src, "run", name="sleeper", args=["30"], timeout=1)
    assert not res.ok and "超时" in res.error


def test_bad_catalog_degrades_without_crash(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("tools: [{", encoding="utf-8")
    src = ToolboxSource(bad)
    assert _call(src, "list").ok is False
    assert asyncio.run(src.list_tools()) == []


def test_specs_risk_marks_active_scan(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, []))
    specs = asyncio.run(src.list_tools())
    names = {s.name: s for s in specs}
    assert set(names) == {"list", "probe", "run", "scan"}
    assert names["list"].risk is Risk.READ
    assert names["run"].risk is Risk.REVERSIBLE
    assert names["scan"].risk is Risk.IRREVERSIBLE


def test_run_refuses_active_scanner_and_scan_refuses_static(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "scanner", "category": "Web", "entry": "echo", "desc": "扫", "risk": "L2", "status": "ok"},
        {"name": "static", "category": "Native", "entry": "echo", "desc": "静态", "risk": "L0", "status": "ok"},
    ]))
    blocked = _call(src, "run", name="scanner", args=["x"])
    assert not blocked.ok and "toolbox.scan" in blocked.error
    wrong = _call(src, "scan", name="static", args=["x"])
    assert not wrong.ok and "toolbox.run" in wrong.error
    allowed = _call(src, "scan", name="scanner", args=["ok"])
    assert allowed.ok and "ok" in allowed.content


def test_entry_alias_resolves(tmp_path: Path) -> None:
    src = ToolboxSource(_catalog(tmp_path, [
        {"name": "Ghidra", "category": "Native", "entry": "echo", "desc": "无头反编译", "risk": "L0", "status": "ok"},
    ]))
    res = _call(src, "run", name="echo", args=["via-entry"])
    assert res.ok and "via-entry" in res.content
