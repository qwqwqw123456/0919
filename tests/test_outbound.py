from __future__ import annotations

import pytest

from doubao_agent.core.outbound import Notifier, OutboundConfig, Outbox, redact


def make_notifier(tmp_path, transport, **kwargs):
    cfg = OutboundConfig(channel="test", **kwargs)
    return Notifier(cfg, tmp_path / "agent.db", transport=transport)


def test_send_once_and_dedupe(tmp_path):
    sent = []
    notifier = make_notifier(tmp_path, sent.append)
    assert notifier.send("开票了")["ok"] is True
    second = notifier.send("开票了")
    assert second["ok"] is False and second["status"] == "duplicate"
    assert sent == ["开票了"]
    notifier.close()


def test_daily_limit(tmp_path):
    sent = []
    notifier = make_notifier(tmp_path, sent.append, daily_limit=2)
    assert notifier.send("a")["ok"] is True
    assert notifier.send("b")["ok"] is True
    blocked = notifier.send("c")
    assert blocked["ok"] is False and blocked["status"] == "limited"
    assert len(sent) == 2
    notifier.close()


def test_failure_is_recorded(tmp_path):
    def boom(text):
        raise RuntimeError("webhook 401")

    notifier = make_notifier(tmp_path, boom)
    result = notifier.send("任务完成")
    assert result["ok"] is False and result["status"] == "failed"
    rows = notifier.outbox.list()
    assert rows[0]["status"] == "failed" and "401" in rows[0]["error"]
    notifier.close()


def test_redact_secrets():
    assert "abcdefgh" not in redact("Authorization: Bearer abcdefghijkl")
    assert "123456789" not in redact("key=sk-123456789")
    assert "********" in redact("token=supersecretvalue")


def test_custom_fingerprint(tmp_path):
    sent = []
    notifier = make_notifier(tmp_path, sent.append)
    assert notifier.send("第一条", fingerprint="fp-1")["ok"] is True
    assert notifier.send("换了文案", fingerprint="fp-1")["status"] == "duplicate"
    assert len(sent) == 1
    notifier.close()


def test_empty_text_not_sent(tmp_path):
    sent = []
    notifier = make_notifier(tmp_path, sent.append)
    assert notifier.send("   ")["status"] == "empty"
    assert sent == []
    notifier.close()
