"""入口分流：纯规则判定，不调用模型，不额外花 token。

明确简单 → 单循环；出现多步、交付物、高风险、要求核验，或拿不准 → 流水线。
保守原则：不确定一律走流水线，漏拦比错放安全。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# 多步信号：这些词出现，通常意味着一件事里套着另一件事
MULTI_STEP_WORDS = (
    "然后", "接着", "再", "分别", "对比", "比较", "逐个", "多个", "批量",
    "汇总", "整理", "修改", "部署", "安装", "导出", "保存", "下载", "上传",
    "执行", "运行", "测试", "调用", "检索", "搜索", "抓取", "爬取", "分析",
    "拆分", "分步", "逐条", "批量处理", "并", "同时",
)

# 交付物信号：要产出一个文件/文档/代码，几乎必然是多步
DELIVERABLE_WORDS = (
    "报告", "表格", "ppt", "文档", "脚本", "代码", "文件", "图片", "视频",
    "清单", "方案", "计划书", "周报", "日报", "作业", "练习", "网页", "网站",
)

# 高风险信号：一旦涉及就不赌单循环。带动作比不带动作危险，所以按"动作模式"匹配。
RISK_PATTERNS = (
    "删除", "删掉", "删了", "清空", "移除", "销毁", "卸载", "覆盖", "格式化",
    "支付", "付款", "转账", "汇款", "下单", "签约", "合同", "法律", "财务",
    "发票", "密钥", "权限", "授权", "吊销", "发布", "上线", "部署", "回滚",
)
# 英文/命令式危险动作，按词边界匹配，避免误伤 mention 之类
RISK_REGEX = re.compile(
    r"(\brm\b|\brm\s+-rf\b|\bdelete\b|\bdrop\s+(table|database)\b|"
    r"\btruncate\b|\bchmod\b|\bchown\b|\btransfer\b|\bwire\b|\bsudo\b)",
    re.IGNORECASE,
)
# "只要说明、不要执行"：这类是知识问答，再怎么带风险字眼也不该进三阶段
EXPLAIN_ONLY = (
    "只说明", "只列出", "只回答", "只评估", "只介绍", "只给", "只写", "只讲", "只解释",
    "仅说明", "仅列出", "仅回答", "仅评估", "只汇报", "只做说明",
)
# 否定语境：这些词出现在动作前，说明是在讲"不要做"
NEGATORS = (
    "不要", "不能", "不用", "不必", "不需要", "无需", "不准", "禁止", "严禁",
    "切勿", "请勿", "勿", "别", "不得",
    "不执行", "不做", "不进行", "不涉及", "不碰", "未执行",
    "do not", "don't", "dont", "never", "without", "no need to",
)
NEGATION_WINDOW = 14

# 用户明确要求核对/证据：单循环做不到自证，直接流水线
VERIFY_WORDS = ("核验", "核对", "校验", "证据", "引用", "溯源", "验证", "比对")

MAX_SIMPLE_CHARS = 40

# 标定分层：样本门控按这五类各要 ≥30 条
STRATA = ("simple_qa", "simple_tool", "multi_step", "high_risk", "image")

STRATUM_LABELS = {
    "simple_qa": "简单问答",
    "simple_tool": "简单工具",
    "multi_step": "多步",
    "high_risk": "高风险",
    "image": "图片",
}


def _negated(text: str, start: int) -> bool:
    """动作前面一小段里出现否定词，就当成"不要做"，不计风险。"""
    prefix = text[max(0, start - NEGATION_WINDOW) : start]
    if any(word in prefix for word in NEGATORS):
        return True
    # "不执行删除""未删除""别删" 这类紧贴动作的单字否定
    return bool(re.search(r"[不未没勿别]$", prefix.rstrip()))


def risk_hit(task: str) -> str:
    """返回命中的高风险动作（已剔除否定语境），没有则空串。"""
    text = task or ""
    for word in RISK_PATTERNS:
        index = text.find(word)
        while index >= 0:
            if not _negated(text, index):
                return word
            index = text.find(word, index + 1)
    for match in RISK_REGEX.finditer(text):
        if not _negated(text, match.start()):
            return match.group(0)
    return ""


def _live(word: str, text: str) -> bool:
    """词出现了，且至少有一处不是"不要做"的语境，才算真信号。"""
    index = text.find(word)
    while index >= 0:
        if not _negated(text, index):
            return True
        index = text.find(word, index + 1)
    return False


def has_risk_signal(task: str) -> bool:
    return bool(risk_hit(task))


def decide_route(
    task: str,
    max_simple_chars: int = MAX_SIMPLE_CHARS,
    has_image: bool = False,
) -> dict[str, Any]:
    """返回 {"route": "single"|"pipeline", "reason": "...", "risk": bool}；不调模型。"""
    text = (task or "").strip()
    if not text:
        return {"route": "single", "reason": "空任务", "risk": False}
    for word in EXPLAIN_ONLY:
        if word in text:
            return {
                "route": "single",
                "reason": f"{word}，只要说明不要执行，走单循环",
                "risk": bool(risk_hit(text)),
            }

    reasons: list[str] = []
    if len(text) > max_simple_chars:
        reasons.append(f"过长({len(text)}字>{max_simple_chars})")
    for word in MULTI_STEP_WORDS:
        if _live(word, text):
            reasons.append(f"多步信号:{word}")
            break
    lowered = text.lower()
    for word in DELIVERABLE_WORDS:
        if _live(word, lowered):
            reasons.append(f"交付物:{word}")
            break
    hit = risk_hit(text)
    if hit:
        reasons.append(f"高风险:{hit}")
    for word in VERIFY_WORDS:
        if word in text:
            reasons.append(f"要求核验:{word}")
            break

    if has_image:
        reasons.append("带图片，拆解会丢图")
    # 唯一命中项只是"文件"、且任务很短（单产物单动作）：允许单循环，硬闸兜底
    if reasons and len(reasons) == 1 and reasons[0] == "交付物:文件" and len(text) <= max_simple_chars:
        return {"route": "single", "reason": "短任务、单文件产物，硬闸兜底", "risk": False}
    if reasons:
        return {"route": "pipeline", "reason": "；".join(reasons), "risk": bool(hit)}
    return {
        "route": "single",
        "reason": "短问答、单产物、无多步与风险信号",
        "risk": False,
    }


def stratum_of(route_info: dict[str, Any], calls: int | None, has_image: bool = False) -> str:
    """把一次运行归到标定分层；分层的目的是让每类都能独立统计。"""
    if has_image:
        return "image"
    if route_info.get("risk"):
        return "high_risk"
    if (route_info.get("decision") or route_info.get("route")) == "pipeline":
        return "multi_step"
    return "simple_tool" if (calls or 0) > 1 else "simple_qa"


def is_sampled(session_id: str, rate: float = 0.1) -> bool:
    """确定性采样：同一会话结论稳定，便于回放时复现。"""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha1((session_id or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000 < int(rate * 10_000)
