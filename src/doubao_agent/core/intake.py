"""人话入口：把一句中文交代拆成可执行的任务卡（不调模型，纯规则）。

用户说「盯着声音玩具北京场开票，有票就提醒我」，不该让人去写 cron。
这里只做一件事：抽出对象 / 条件 / 频率 / 动作 / 确认级 / 到期，缺什么就明说缺什么，
拆不准就把原话和已识别项一起交出去，绝不编造。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

CONFIRM_KEYWORDS = {
    2: ("下单", "购买", "付款", "支付", "转账", "充值", "发消息", "发邮件", "打电话", "退订", "预订并支付"),
    1: ("预约", "预订", "订座", "订餐", "订", "报名", "提交", "发送", "办理", "申请", "改签", "取消订单"),
}

ACTION_KEYWORDS = (
    ("buy", ("下单", "购买", "买一个", "买台", "买瓶")),
    ("send", ("发消息", "发邮件", "发给他", "发给")),
    ("call", ("打电话", "致电")),
    ("remind", ("提醒", "通知", "告诉我", "推送", "喊我")),
)

_WEEKDAYS = {
    "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6, "周天": 6,
    "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
}

_FILLER = re.compile(r"^(帮我|帮忙|麻烦|请|你|给我|替我|盯着|盯一下|关注|跟进|记得|到时候)+")


@dataclass
class TaskSpec:
    """一张任务卡。cron 可直接喂给现有 scheduler。"""

    title: str
    target: str
    condition: str
    cron: str
    action: str
    confirm_level: int
    due_at: float | None = None
    source_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "target": self.target,
            "condition": self.condition,
            "cron": self.cron,
            "action": self.action,
            "confirm_level": self.confirm_level,
            "due_at": self.due_at,
            "source_text": self.source_text,
            "warnings": list(self.warnings),
        }

    def describe(self) -> str:
        """写进 schedule.task 的完整描述，保留原话，便于以后追溯。"""
        due = datetime.fromtimestamp(self.due_at).strftime("%Y-%m-%d %H:%M") if self.due_at else "长期"
        return (
            f"【任务卡】对象={self.target}；条件={self.condition}；动作={self.action}；"
            f"确认级={self.confirm_level}；到期={due}；原话={self.source_text}"
        )


def _pick_confirm_level(text: str) -> tuple[int, list[str]]:
    hits: list[str] = []
    for level in (2, 1):
        for word in CONFIRM_KEYWORDS[level]:
            if word in text:
                hits.append(word)
                return level, hits
    return 0, hits


def _pick_action(text: str) -> str:
    for action, words in ACTION_KEYWORDS:
        if any(w in text for w in words):
            return action
    return "push"


def _parse_frequency(text: str) -> tuple[str, str]:
    """返回 (cron, 人类可读频率)。"""
    m = re.search(r"每(?:隔)?\s*(\d+)\s*(?:个)?(小时|h|H|钟头)", text)
    if m:
        n = max(1, min(23, int(m.group(1))))
        if n == 1:
            return "0 * * * *", "每小时"
        return f"0 */{n} * * *", f"每{n}小时"

    m = re.search(r"每(?:隔)?\s*(\d+)\s*(?:分钟|min)", text)
    if m:
        n = max(5, min(59, int(m.group(1))))
        return f"*/{n} * * *", f"每{n}分钟"

    if re.search(r"每小时|每个小时|每小时一次", text):
        return "0 * * * *", "每小时"

    if re.search(r"每周|每个星期", text):
        weekday = next((idx for name, idx in _WEEKDAYS.items() if name in text), None)
        day = weekday if weekday is not None else 0
        return f"10 9 * * {day}", "每周"

    if re.search(r"每天|每日|天天|每早|每天早上", text):
        tod = _parse_time_of_day(text)
        if tod:
            return f"{tod[1]} {tod[0]} * * *", f"每天 {tod[0]:02d}:{tod[1]:02d}"
        if re.search(r"早上|上午", text):
            return "40 7 * * *", "每天早上"
        if re.search(r"晚上|夜里", text):
            return "40 20 * * *", "每天晚上"
        return "40 8 * * *", "每天"

    if re.search(r"开票|抢票|降价|补货|券|到期|放票", text):
        return "0 */6 * * *", "每6小时（盯梢类默认）"
    return "0 */6 * * *", "每6小时（默认）"


def _parse_time_of_day(text: str) -> tuple[int, int] | None:
    m = re.search(
        r"(凌晨|早上|上午|中午|下午|傍晚|晚上)?\s*(\d{1,2})\s*[点:：]\s*(?:(\d{1,2})\s*分|(\d{2})\b)?",
        text,
    )
    if not m:
        return None
    hour = int(m.group(2))
    minute = int(m.group(3) or m.group(4) or 0)
    period = m.group(1) or ""
    if period in ("下午", "傍晚", "晚上") and hour < 12:
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _parse_due(text: str, now: float | None = None) -> tuple[float | None, str]:
    now = time.time() if now is None else now
    base = datetime.fromtimestamp(now)
    tod = _parse_time_of_day(text)

    def at(day: datetime, _label: str = "") -> float:
        h, m = tod if tod else (23, 59)
        return day.replace(hour=h, minute=m, second=0, microsecond=0).timestamp()

    if "后天" in text:
        return at(base + timedelta(days=2)), "后天"
    if "明天" in text or "明日" in text:
        return at(base + timedelta(days=1)), "明天"
    if "今天" in text or "今日" in text or "今晚" in text:
        return at(base), "今天"

    m = re.search(r"(\d+)\s*天(?:后|以后|之后)", text)
    if m:
        return at(base + timedelta(days=int(m.group(1)))), f"{m.group(1)}天后"

    m = re.search(r"(\d+)\s*(?:个)?(?:小时|钟头)(?:后|以后|之后)", text)
    if m:
        return now + int(m.group(1)) * 3600, f"{m.group(1)}小时后"

    m = re.search(r"(\d+)\s*(?:分钟)(?:后|以后|之后)", text)
    if m:
        return now + int(m.group(1)) * 60, f"{m.group(1)}分钟后"

    for name, idx in _WEEKDAYS.items():
        if name in text:
            delta = (idx - base.weekday()) % 7 or 7
            return at(base + timedelta(days=delta)), f"下周{name}"

    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = base.year + (1 if month < base.month else 0)
        try:
            target = base.replace(year=year, month=month, day=day)
        except ValueError:
            return None, "日期无法解析"
        if tod:
            target = target.replace(hour=tod[0], minute=tod[1], second=0, microsecond=0)
        return target.timestamp(), f"{month}月{day}日"

    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        try:
            target = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None, "日期无法解析"
        if tod:
            target = target.replace(hour=tod[0], minute=tod[1])
        return target.timestamp(), target.strftime("%Y-%m-%d")

    if tod:
        target = at(base)
        if target <= now:
            target = at(base + timedelta(days=1))
        return target, f"{tod[0]:02d}:{tod[1]:02d}"
    return None, ""


def _extract_condition(text: str) -> str:
    """条件是被动作的触发句：中文里几乎都写成「条件 + 就 + 动作」。"""
    for clause in re.split(r"[，,。；;]", text):
        if "就" in clause:
            cond = clause.split("就", 1)[0].strip()
            cond = re.sub(r"^(如果|若是|要是|当|一旦|等|等到|然后)", "", cond).strip()
            if len(cond) >= 2:
                return cond
    for sep in ("如果有", "如果", "一旦", "等到"):
        if sep in text:
            tail = text.split(sep, 1)[1].strip(" ，。！")
            if tail:
                return tail.split("就")[0].strip() or tail
    return ""


def _extract_target(text: str) -> str:
    quoted = re.findall(r"「(.+?)」|\"(.+?)\"|“(.+?)”|'(.+?)'", text)
    for group in quoted:
        value = next((g for g in group if g), "")
        if value:
            return value.strip()

    core = re.split(r"[，,。；;]", text)[0]
    core = re.sub(r"\d+\s*个?人", "", core)
    core = re.sub(r"每天|每日|每周|每\s*\d+\s*(?:小时|分钟|分钟)|明天|后天|今天|今晚|早上|上午|中午|下午|傍晚|晚上|凌晨", "", core)
    core = re.sub(r"\d{1,2}\s*[点:：]\s*\d{0,2}\s*分?", "", core)
    segments = [core]
    for word in ("提醒我", "通知我", "告诉我", "盯着", "盯一下", "跟进", "关注", "看看", "查查", "帮忙看", "订个", "订", "买", "发消息", "打电话"):
        if word in core:
            before, after = core.split(word, 1)
            segments.extend([before, after])
    # 动作词后面那段才是"要办的事"，优先用它；没有再退回最长片段
    tail = ""
    for word in ("提醒我", "通知我", "告诉我", "盯着", "盯一下", "跟进", "关注", "看看", "查查", "订个", "订", "买", "发消息", "打电话"):
        if word in core and core.split(word, 1)[1].strip():
            tail = core.split(word, 1)[1]
            break
    pool = [tail] if tail else segments
    best = ""
    for seg in pool:
        seg = _FILLER.sub("", seg.strip())
        seg = re.sub(r"^(看看|查查|瞅瞅|一下|下)", "", seg).strip(" ，。的地了")
        seg = re.sub(r"(快没了没|没了没|还有没有|有没有|吗)$", "", seg).strip()
        if len(seg) > len(best):
            best = seg
    return best[:60]


def parse_task(text: str, now: float | None = None) -> TaskSpec:
    """一句话 → 任务卡。缺字段不补假值，写进 warnings。"""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("需要一句话，比如：盯着某场演出开票，有票就提醒我")

    cron, freq_label = _parse_frequency(raw)
    due, due_label = _parse_due(raw, now)
    action = _pick_action(raw)
    level, level_hits = _pick_confirm_level(raw)
    target = _extract_target(raw)
    condition = _extract_condition(raw)

    warnings: list[str] = []
    if not target:
        warnings.append("没识别到监控对象，补一句「盯什么」")
    if not condition and action == "push":
        condition = "有新进展就推送"
        warnings.append("没识别到触发条件，默认按「有新进展就推送」")
    if action == "call":
        warnings.append("代打电话无外呼能力，只能给号码与话术，需人工拨出")
    if level >= 2:
        warnings.append(f"含「{'/'.join(level_hits)}」，下单/外发前必须你点头")

    title = (target or raw[:16])[:24]
    return TaskSpec(
        title=title,
        target=target or raw[:60],
        condition=condition or "有新进展就推送",
        cron=cron,
        action=action,
        confirm_level=level,
        due_at=due,
        source_text=raw,
        warnings=warnings,
    )


def explain(spec: TaskSpec) -> str:
    """人话回读：先说我理解成什么，再问缺的那一句。"""
    due = datetime.fromtimestamp(spec.due_at).strftime("%m-%d %H:%M") if spec.due_at else "长期"
    lines = [
        f"我理解成：{spec.title}",
        f"· 盯：{spec.target}",
        f"· 条件：{spec.condition}",
        f"· 频率：{spec.cron}  到期：{due}",
        f"· 命中后：{spec.action}　确认级：{spec.confirm_level}",
    ]
    if spec.warnings:
        lines.append("· 注意：" + "；".join(spec.warnings))
    return "\n".join(lines)
