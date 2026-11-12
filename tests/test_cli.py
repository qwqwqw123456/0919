from __future__ import annotations

import json

from doubao_agent.cli import main
from doubao_agent.core.audit import AuditLog


def _write_config(tmp_path) -> str:
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        f"workspace: {tmp_path}\nruns_dir: {tmp_path / 'runs'}\n"
        f"db_path: {tmp_path / '.agent' / 'agent.db'}\nport: 8766\n",
        encoding="utf-8",
    )
    return str(cfg_path)


def test_cli_audit_verify_ok(tmp_path, capsys) -> None:
    cfg_path = _write_config(tmp_path)
    log = AuditLog(tmp_path / "runs")
    log.append("s1", "start", task="查周报")
    log.append("s1", "finish", status="completed")

    assert main(["--config", cfg_path, "audit", "verify"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "s1" in out and "2 条" in out


def test_cli_audit_verify_fails_on_tamper(tmp_path, capsys) -> None:
    cfg_path = _write_config(tmp_path)
    log = AuditLog(tmp_path / "runs")
    log.append("s1", "start", task="原始任务")

    path = tmp_path / "runs" / "s1" / "audit.jsonl"
    record = json.loads(path.read_text("utf-8").strip())
    record["task"] = "篡改"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    assert main(["--config", cfg_path, "audit", "verify", "s1"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_audit_sessions(tmp_path, capsys) -> None:
    cfg_path = _write_config(tmp_path)
    log = AuditLog(tmp_path / "runs")
    log.append("abc", "start")

    assert main(["--config", cfg_path, "audit", "sessions"]) == 0
    assert "abc" in capsys.readouterr().out
