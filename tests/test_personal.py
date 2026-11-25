from __future__ import annotations

import time

import pytest

from doubao_agent.core.personal import (
    ConfirmRequired,
    PersonalStore,
    build_briefing,
    check_gate,
)


@pytest.fixture()
def store(tmp_path):
    store = PersonalStore(tmp_path / "agent.db")
    yield store
    store.close()


def test_add_and_list_task(store):
    task_id = store.add(
        name="盯开票",
        target="声音玩具乐队 北京 场次",
        condition="官方票务页出现可购状态",
        freq="6h",
        confirm_level=0,
        due_at=time.time() + 7 * 86400,
    )
    assert task_id > 0
    rows = store.list()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "active"
    assert row["target"] == "声音玩具乐队 北京 场次"
    assert store.get(task_id)["name"] == "盯开票"


def test_add_rejects_incomplete_card(store):
    with pytest.raises(ValueError):
        store.add(name="", target="x", condition="y")
    with pytest.raises(ValueError):
        store.add(name="a", target=" ", condition="y")
    with pytest.raises(ValueError):
        store.add(name="a", target="x", condition="")
    with pytest.raises(ValueError):
        store.add(name="a", target="x", condition="y", confirm_level=9)


def test_status_transitions_and_filter(store):
    task_id = store.add(name="a", target="x", condition="y")
    assert store.set_status(task_id, "paused") is True
    assert [r["status"] for r in store.list("paused")] == ["paused"]
    assert store.list("active") == []
    assert store.set_status(task_id, "done") is True
    with pytest.raises(ValueError):
        store.set_status(task_id, "unknown")
    assert store.set_status(999, "done") is False


def test_touch_records_result(store):
    task_id = store.add(name="a", target="x", condition="y")
    assert store.touch(task_id, "无命中") is True
    row = store.get(task_id)
    assert row["last_result"] == "无命中"
    assert row["last_check"] is not None


def test_push_log(store):
    task_id = store.add(name="a", target="x", condition="y")
    store.log_push(task_id, "开到 380 元档")
    store.log_push(task_id, "二次命中")
    pushes = store.pushes(5)
    assert pushes[0]["summary"] == "二次命中"
    assert len(pushes) == 2


def test_briefing_empty_is_one_line():
    lines = build_briefing([])
    assert lines == ["今天无事：没有在跟进的事，也没有命中记录"]


def test_briefing_with_hit_and_task(store):
    task_id = store.add(name="盯开票", target="x", condition="y")
    store.log_push(task_id, "开票页已上架")
    lines = build_briefing(store.list(), store.pushes(5))
    assert lines[0].startswith("在跟进 1 条")
    assert any("今日命中" in line and "开票页已上架" in line for line in lines)


def test_briefing_marks_due_and_blocked():
    now = time.time()
    tasks = [
        {"id": 1, "name": "补货", "status": "active", "due_at": now + 3600, "confirm_level": 2},
        {"id": 2, "name": "降价", "status": "active", "due_at": None, "confirm_level": 0},
    ]
    lines = build_briefing(tasks, [], now=now)
    assert any("临期" in line and "补货" in line for line in lines)
    assert any("待你点头" in line for line in lines)


def test_briefing_caps_lines():
    tasks = [
        {
            "id": i,
            "name": f"任务{i}",
            "status": "active",
            "due_at": time.time() + 60,
            "confirm_level": 1,
            "last_result": "无命中",
        }
        for i in range(5)
    ]
    lines = build_briefing(tasks, [], max_lines=3)
    assert len(lines) == 3


def test_gate_levels():
    check_gate(0)
    with pytest.raises(ConfirmRequired):
        check_gate(1)
    with pytest.raises(ConfirmRequired):
        check_gate(2)
    check_gate(1, user_confirmed=True)
    check_gate(2, user_confirmed=True)
    with pytest.raises(ValueError):
        check_gate(7)
