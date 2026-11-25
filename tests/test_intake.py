from __future__ import annotations

import time

import pytest

from doubao_agent.core.intake import TaskSpec, explain, parse_task


def test_concert_ticket_intent():
    spec = parse_task("盯着声音玩具北京场开票，有票就提醒我")
    assert isinstance(spec, TaskSpec)
    assert "开票" in spec.target
    assert spec.condition == "有票"
    assert spec.cron == "0 */6 * * *"
    assert spec.action == "remind"
    assert spec.confirm_level == 0
    assert spec.warnings == []


def test_daily_at_four_pm():
    spec = parse_task("每天下午4点提醒我发最终版资料")
    assert spec.cron == "0 16 * * *"
    due = time.localtime(spec.due_at)
    assert (due.tm_hour, due.tm_min) == (16, 0)


def test_relative_due_and_time_of_day_not_misread():
    spec = parse_task("帮我订明天晚上7点4个人的餐厅，订好告诉我")
    due = time.localtime(spec.due_at)
    assert (due.tm_hour, due.tm_min) == (19, 0)
    assert spec.confirm_level == 1
    assert "餐厅" in spec.target


def test_three_days_later():
    now = time.time()
    spec = parse_task("3天后提醒我交房租", now=now)
    # 没给钟点就落到当天 23:59，因此只校验日期区间
    assert spec.due_at >= now + 3 * 86400 - 3600
    assert spec.due_at <= now + 4 * 86400
    assert time.localtime(spec.due_at).tm_hour == 23


def test_purchase_needs_confirmation():
    spec = parse_task("盯着这个商品降价，降到399就提醒我，降价了下单")
    assert spec.confirm_level == 2
    assert spec.action == "buy"
    assert any("点头" in w for w in spec.warnings)


def test_call_action_is_flagged():
    spec = parse_task("帮我打电话问一下餐厅有没有包间")
    assert spec.action == "call"
    assert any("外呼" in w for w in spec.warnings)


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        parse_task("   ")


def test_explain_reads_back_four_fields():
    text = explain(parse_task("每2小时帮我看看猫粮，快没了就提醒我买"))
    for field in ("盯：", "条件：", "频率：", "命中后："):
        assert field in text
