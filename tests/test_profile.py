from __future__ import annotations

import pytest

from doubao_agent.core.profile import ProfileStore


@pytest.fixture()
def store(tmp_path):
    store = ProfileStore(tmp_path / "agent.db")
    yield store
    store.close()


def test_set_and_get(store):
    entry = store.set("称呼", "老板", source="user")
    assert entry["value"] == "老板"
    assert entry["status"] == "active"
    assert store.get("称呼")["confidence"] == 1.0


def test_conflict_keeps_history(store):
    store.set("作息", "早上7点起")
    store.set("作息", "早上8点起")
    assert store.get("作息")["value"] == "早上8点起"
    history = store.history("作息")
    assert history[0]["old_value"] == "早上7点起"
    assert history[0]["new_value"] == "早上8点起"


def test_forget_excludes_from_context(store):
    store.set("称呼", "老板")
    store.set("禁忌", "别叫我老师")
    assert "老板" in store.context_text()
    assert store.forget("禁忌") is True
    assert "别叫我老师" not in store.context_text()
    assert store.get("禁忌")["status"] == "forgotten"
    assert store.forget("不存在") is False


def test_rejects_secrets(store):
    with pytest.raises(ValueError):
        store.set("api_key", "sk-123")
    with pytest.raises(ValueError):
        store.set("密码", "123456")


def test_empty_values_rejected(store):
    with pytest.raises(ValueError):
        store.set("称呼", "  ")


def test_context_text_empty_and_truncated(tmp_path):
    empty = ProfileStore(tmp_path / "empty.db")
    assert empty.context_text() == ""
    empty.close()

    store = ProfileStore(tmp_path / "many.db")
    for i in range(12):
        store.set(f"偏好{i}", "很长的一条偏好描述" * 3)
    text = store.context_text(limit=12, max_chars=200)
    assert text.startswith("关于用户（画像，按此口径）：")
    assert len(text) <= 201
    store.close()
