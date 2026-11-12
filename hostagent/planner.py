"""契约规划层：只允许编出 interface-manifest 白名单内的节点。

本地是规则层（manifest 第 E 层），不联网、不调模型。真正的意图理解由宿主
dag_planner 负责；这里提供等价的确定性实现，保证离线可验证、无幻觉。
"""

from __future__ import annotations

from typing import Any

from .contract import NODES, get_node
from .models import Plan, PlanNode


class ContractError(ValueError):
    """计划违反契约（幻觉工具 / 非法前驱 / 环 / 重复 id）。"""


def validate(plan: Plan) -> None:
    if not plan.nodes:
        raise ContractError("计划为空")
    seen: dict[str, PlanNode] = {}
    for node in plan.nodes:
        if node.id in seen:
            raise ContractError(f"节点 id 重复: {node.id}")
        spec = get_node(node.tool)
        if spec is None:
            raise ContractError(f"幻觉工具不在契约白名单: {node.tool}")
        for dep in node.needs:
            if dep not in seen:
                raise ContractError(f"节点 {node.id} 的前驱 {dep} 不存在或未在前")
            dep_spec = get_node(seen[dep].tool)
            assert dep_spec is not None
            if spec.needs and dep_spec.tool not in spec.needs:
                raise ContractError(
                    f"节点 {node.id}({node.tool}) 不接受前驱 {seen[dep].tool}；"
                    f"合法前驱: {'/'.join(spec.needs)}"
                )
            if not spec.needs:
                raise ContractError(f"节点 {node.id}({node.tool}) 不接受任何前驱")
        seen[node.id] = node


def plan_from_spec(spec: dict[str, Any]) -> Plan:
    """从显式 spec 构造计划并校验（对应 pipeline.py --spec）。"""
    plan = Plan(
        intent=str(spec.get("intent", "")),
        nodes=[PlanNode.from_dict(n) for n in spec.get("nodes", [])],
    )
    validate(plan)
    return plan


def _n(tool: str, params: dict[str, Any] | None = None, needs: list[str] | None = None) -> PlanNode:
    return PlanNode(id=tool, tool=tool, params=params or {}, needs=needs or [])


def plan_by_rules(intent: str) -> Plan:
    """确定性规则规划。命中不到模板时退回「记忆召回 + 格式化」最小链。"""
    text = intent.strip()
    lower = text.lower()

    def has(*keys: str) -> bool:
        return any(k in text or k in lower for k in keys)

    if has("周报", "月报", "汇报"):
        nodes = [
            _n("search_web", {"query": "{intent}", "top_n": 3}),
            _n("read_many", {"urls": "{search_web}", "max_chars": 4000}, ["search_web"]),
            _n("create_doc", {"title": "{intent}", "content": "{summary}"}, ["read_many"]),
        ]
    elif has("表格", "excel", "xlsx", "清单"):
        nodes = [
            _n("search_web", {"query": "{intent}", "top_n": 3}),
            _n("read_many", {"urls": "{search_web}", "max_chars": 4000}, ["search_web"]),
            _n("create_sheet", {"title": "{intent}", "header": ["来源", "摘要"], "rows": []},
               ["read_many"]),
        ]
    elif has("ppt", "幻灯片", "演示"):
        nodes = [
            _n("search_web", {"query": "{intent}", "top_n": 3}),
            _n("read_many", {"urls": "{search_web}", "max_chars": 4000}, ["search_web"]),
            _n("create_ppt", {"title": "{intent}", "slides": []}, ["read_many"]),
        ]
    elif has("发消息", "通知", "告警", "告诉"):
        nodes = [
            _n("memory_retrieve", {"query": "{intent}", "k": 3}),
            _n("send_message", {"text": "{summary}"}, ["memory_retrieve"]),
        ]
    elif has("日程", "会议", "日历", "约"):
        nodes = [
            _n("memory_retrieve", {"query": "{intent}", "k": 3}),
            _n("create_calendar", {"title": "{intent}", "meeting_time": "", "meeting_end": "",
                                   "description": "{summary}"}, ["memory_retrieve"]),
        ]
    elif has("法条", "法律", "法规"):
        nodes = [_n("legal_search", {"query": "{intent}"})]
    elif has("股票", "行情", "财报", "基金", "汇率"):
        nodes = [
            _n("finance_search", {"query": "{intent}"}),
            _n("finance_lookup", {"query": "{intent}"}),
        ]
    elif has("工商", "公司", "企业"):
        nodes = [_n("company_search", {"query": "{intent}"})]
    elif has("路线", "怎么走", "地铁", "公交"):
        nodes = [_n("map_route", {"query": "{intent}"})]
    elif has("代码", "仓库", "github", "pr"):
        nodes = [_n("github_code", {"query": "{intent}"})]
    elif has("查资料", "研究", "调研", "搜索", "最新"):
        nodes = [
            _n("search_web", {"query": "{intent}", "top_n": 3}),
            _n("read_many", {"urls": "{search_web}", "max_chars": 4000}, ["search_web"]),
            _n("format_text", {"text": "{summary}", "max_chars": 6000}),
        ]
    else:
        nodes = [
            _n("memory_retrieve", {"query": "{intent}", "k": 3}),
            _n("format_text", {"text": "{summary}", "max_chars": 6000}),
        ]

    plan = Plan(intent=text, nodes=nodes)
    validate(plan)
    return plan


def describe_contract() -> list[dict[str, Any]]:
    """给 CLI / 控制台用的契约总览。"""
    return [
        {
            "tool": n.tool,
            "delegate": n.delegate,
            "action": n.action,
            "risk": n.risk.value,
            "approval": n.approval.value,
            "produces": n.produces,
            "needs": list(n.needs),
            "note": n.note,
        }
        for n in NODES.values()
    ]
