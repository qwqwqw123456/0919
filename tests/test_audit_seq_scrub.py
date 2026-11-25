from __future__ import annotations

import json

from doubao_agent.core.audit import AuditLog, redact


def test_seq_increments_and_enters_chain(tmp_path):
    log = AuditLog(tmp_path)
    records = [log.append("s", "step", i=i) for i in range(3)]
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert log.verify("s") == (True, 3)


def test_tampered_seq_detected(tmp_path):
    log = AuditLog(tmp_path)
    log.append("s", "a")
    log.append("s", "b")
    path = log._path("s")
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[1]["seq"] = 5
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    assert log.verify("s") == (False, 1)


def test_inline_secret_in_code_is_masked(tmp_path):
    log = AuditLog(tmp_path)
    record = log.append(
        "s",
        "tool_run",
        code="curl -H 'Authorization: Bearer abcdefgh123456' -d api_key=abcdefg",
        command="export TOKEN=supersecret",
        content="sk-abcdefghijklmn 和 eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abcd1234",
    )
    raw = json.dumps(record, ensure_ascii=False)
    for leaked in ("abcdefgh123456", "abcdefg", "supersecret", "abcdefghijklmn", "abcd1234"):
        assert leaked not in raw
    assert "bearer ***" in record["code"].lower()
    assert log.verify("s") == (True, 1)


def test_plain_prose_not_over_scrubbed():
    text = "把 token 放哪都行，password 记得改"
    assert redact(text) == text  # 没有 = 或 : 就不动，避免把正文打坏
