from __future__ import annotations

import pytest

from doubao_agent.core.knowledge import KnowledgeStore
from doubao_agent.core.longterm import LongTermMemory
from doubao_agent.core.working import WorkingMemory


def test_working_memory_session_lifecycle(tmp_path) -> None:
    db = tmp_path / "agent.db"
    working = WorkingMemory(db)
    assert working.load()["初始化阶段"] == "cold"

    state = working.begin_session("查一下豆包云电脑的周报")
    assert state["会话计数"] == 1
    assert state["焦点线"].startswith("查一下豆包")
    assert state["工作流节点"] == "运行中"

    working.end_session("completed", "已生成周报")
    reopened = WorkingMemory(db)
    saved = reopened.load()
    assert saved["工作流节点"] == "completed"
    assert saved["待回顾"] == "已生成周报"
    assert saved["会话计数"] == 1

    again = reopened.begin_session("第二轮")
    assert again["会话计数"] == 2
    assert again["上次会话日期"] == saved["时间戳"]


def test_working_memory_rejects_unknown_field_and_respects_lock(tmp_path) -> None:
    working = WorkingMemory(tmp_path / "agent.db")
    with pytest.raises(ValueError):
        working.update(不存在的字段="x")

    with working.locked():
        assert working.load()["写入锁"] == "locked"
        with pytest.raises(RuntimeError):
            working.update(焦点线="抢写")
    assert working.load()["写入锁"] == "unlocked"

    working.update(焦点线="新焦点", 我的名字="豆包")
    text = working.context_text()
    assert "焦点线：新焦点" in text and "我的名字：豆包" in text
    assert "写入锁" not in text


def test_longterm_three_tables(tmp_path) -> None:
    store = LongTermMemory(tmp_path / "agent.db")
    store.add_lesson(
        "发布失败",
        judgment="没做回滚演练",
        outcome="回滚耗时 40 分钟",
        lesson="发布前必须演练回滚",
        date_str="2026-11-01",
    )
    store.add_observation(
        "接口延迟翻倍", observation="高峰期明显", cause="缺索引", followup="补索引"
    )
    store.add_deviation("100", "40", cause="预算误估", fix="重新评估")

    lesson = store.lessons()[0]
    assert lesson["date"] == "2026-11-01" and lesson["event"] == "发布失败"
    observation = store.observations()[0]
    assert observation["phenomenon"] == "接口延迟翻倍" and observation["cause"] == "缺索引"
    deviation = store.deviations()[0]
    assert deviation["gap"] == "+60"
    assert store.stats() == {"lessons": 1, "observations": 1, "deviations": 1}


def test_knowledge_domains_index_and_filter(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "agent.db")
    store.add_text("发布复盘", "发布流程必须演练回滚", domain="02 自我成长")
    store.add_text("零散资料", "普通的工具说明")

    docs = store.list_documents()
    assert docs[0]["domain"] == ""
    assert docs[1]["domain"] == "02"

    stats = {item["code"]: item["documents"] for item in store.domain_stats()}
    assert stats["02"] == 1 and stats["01"] == 0

    hits = store.search("发布流程", domain="02")
    assert hits and all(hit["domain"] == "02" for hit in hits)
    assert store.search("发布流程", domain="01") == []

    with pytest.raises(ValueError):
        store.add_text("坏域", "内容", domain="99 不存在")
