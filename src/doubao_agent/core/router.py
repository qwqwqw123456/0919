"""入口分流：纯规则判定，不调用模型，不额外花 token。

明确简单 → 单循环；出现多步、交付物、高风险、要求核验，或拿不准 → 流水线。
保守原则：不确定一律走流水线，漏拦比错放安全。
"""

from __future__ import annotations

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

# 高风险信号：一旦涉及就不赌单循环
RISK_WORDS = (
    "删除", "删掉", "移除", "支付", "付款", "转账", "下单", "合同", "法律",
    "财务", "发票", "密钥", "权限", "发布", "上线", "部署", "覆盖", "格式化",
)

# 用户明确要求核对/证据：单循环做不到自证，直接流水线
VERIFY_WORDS = ("核验", "核对", "校验", "证据", "引用", "溯源", "验证", "比对")

MAX_SIMPLE_CHARS = 40


def decide_route(task: str, max_simple_chars: int = MAX_SIMPLE_CHARS) -> dict[str, str]:
    """返回 {"route": "single"|"pipeline", "reason": "..."}；不调模型。"""
    text = (task or "").strip()
    if not text:
        return {"route": "single", "reason": "空任务"}

    reasons: list[str] = []
    if len(text) > max_simple_chars:
        reasons.append(f"过长({len(text)}字>{max_simple_chars})")
    for word in MULTI_STEP_WORDS:
        if word in text:
            reasons.append(f"多步信号:{word}")
            break
    lowered = text.lower()
    for word in DELIVERABLE_WORDS:
        if word in lowered:
            reasons.append(f"交付物:{word}")
            break
    for word in RISK_WORDS:
        if word in text:
            reasons.append(f"高风险:{word}")
            break
    for word in VERIFY_WORDS:
        if word in text:
            reasons.append(f"要求核验:{word}")
            break

    if reasons:
        return {"route": "pipeline", "reason": "；".join(reasons)}
    return {"route": "single", "reason": "短问答、单产物、无多步与风险信号"}
