from __future__ import annotations

import json

from doubao_agent.core.audit import AuditLog


def test_audit_redacts_secrets(tmp_path) -> None:
    log = AuditLog(tmp_path)
    log.append(
        "s1",
        "tool",
        tool="send_message",
        arguments={
            "token": "abc",
            "api_key": "k-123",
            "access_token": "t",
            "target_id": "user-1",
            "max_tokens": 5,
        },
    )

    row = log.read("s1")[0]
    arguments = row["arguments"]
    assert arguments["token"] == "***"
    assert arguments["api_key"] == "***"
    assert arguments["access_token"] == "***"
    assert arguments["target_id"] == "user-1"
    # max_tokens 不是凭据，不能被误伤
    assert arguments["max_tokens"] == 5
    assert "abc" not in json.dumps(row)


def test_audit_verify_passes_and_detects_tampering(tmp_path) -> None:
    log = AuditLog(tmp_path)
    log.append("s1", "start", task="查周报")
    log.append("s1", "tool", tool="knowledge.search")
    log.append("s1", "finish", status="completed")

    assert log.verify("s1") == (True, 3)

    path = tmp_path / "s1" / "audit.jsonl"
    lines = path.read_text("utf-8").splitlines()
    record = json.loads(lines[0])
    record["task"] = "被改过的任务"
    lines[0] = json.dumps(record, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, count = log.verify("s1")
    assert ok is False and count == 0


def test_audit_sessions_lists_dirs_with_audit(tmp_path) -> None:
    log = AuditLog(tmp_path)
    log.append("alpha", "start")
    log.append("beta", "start")
    (tmp_path / "gamma").mkdir()

    assert set(log.sessions()) == {"alpha", "beta"}
    assert log.verify("missing") == (True, 0)
