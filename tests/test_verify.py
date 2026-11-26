from __future__ import annotations

import json
from pathlib import Path

from doubao_agent.core.verify import (
    check_exit,
    check_file,
    check_text,
    check_claimed_paths,
    paths_in,
    run_specs,
    summarize,
)


def test_check_file_requires_existing_non_empty(tmp_path: Path) -> None:
    assert check_file(tmp_path / "missing.md")["ok"] is False
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    assert check_file(empty)["ok"] is False
    good = tmp_path / "good.md"
    good.write_text("结论：4", encoding="utf-8")
    assert check_file(good, contains=["结论"])["ok"] is True


def test_check_file_json_parse(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert check_file(bad, must_parse_json=True)["ok"] is False
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert check_file(ok, must_parse_json=True)["ok"] is True


def test_check_text_and_exit() -> None:
    assert check_text("结果是 4", contains=["4"])["ok"] is True
    assert check_text("结果是 4", not_contains=["5"])["ok"] is True
    assert check_text("结果是 4", regex=r"\d")["ok"] is True
    assert check_text("结果是 4", contains=["5"])["ok"] is False
    assert check_exit(0)["ok"] is True
    assert check_exit(1)["ok"] is False
    assert check_exit(None)["ok"] is False


def test_only_claimed_writes_get_disk_check(tmp_path: Path) -> None:
    target = tmp_path / "out.md"
    # 只提到路径、没说写：不查磁盘
    assert check_claimed_paths([{"output": f"参考 {target}"}]) == []
    target.write_text("内容", encoding="utf-8")
    checks = check_claimed_paths([{"output": f"已写入 {target}"}])
    assert checks and checks[0]["ok"] is True


def test_paths_in_dedupes() -> None:
    text = "写到 /tmp/a.md，再复制 /tmp/a.md"
    assert paths_in(text) == ["/tmp/a.md"]


def test_run_specs_and_summarize(tmp_path: Path) -> None:
    good = tmp_path / "x.txt"
    good.write_text("hello", encoding="utf-8")
    checks = run_specs(
        [
            {"kind": "file", "path": str(good), "contains": ["hello"]},
            {"kind": "text", "value": "hello", "contains": ["hello"]},
            {"kind": "exit", "code": 0},
            {"kind": "file", "path": str(tmp_path / "nope.txt")},
        ]
    )
    result = summarize(checks)
    assert result["pass"] is False
    assert any("nope" in issue for issue in result["issues"])
    assert summarize(checks[:3])["pass"] is True
