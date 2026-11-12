import pytest

from hostagent.models import Plan, PlanNode
from hostagent.planner import ContractError, plan_by_rules, plan_from_spec, validate


def test_weekly_report_plan_is_contract_valid():
    plan = plan_by_rules("帮我写一份周报")
    validate(plan)
    tools = [n.tool for n in plan.nodes]
    assert tools == ["search_web", "read_many", "create_doc"]
    assert plan.node("read_many").needs == ["search_web"]
    assert plan.node("create_doc").needs == ["read_many"]


def test_hallucinated_tool_is_rejected():
    plan = Plan("x", [PlanNode("evil", "delete_everything")])
    with pytest.raises(ContractError, match="幻觉工具"):
        validate(plan)


def test_illegal_predecessor_is_rejected():
    plan = Plan("x", [
        PlanNode("read_web", "read_web"),
        PlanNode("read_many", "read_many", needs=["read_web"]),
    ])
    with pytest.raises(ContractError, match="不接受前驱"):
        validate(plan)


def test_external_connector_rejects_predecessor():
    plan = Plan("x", [
        PlanNode("format_text", "format_text"),
        PlanNode("legal_search", "legal_search", needs=["format_text"]),
    ])
    with pytest.raises(ContractError, match="不接受任何前驱"):
        validate(plan)


def test_spec_roundtrip():
    spec = {
        "intent": "算个东西",
        "nodes": [
            {"id": "format_text", "tool": "format_text", "params": {"text": "hi"}},
        ],
    }
    plan = plan_from_spec(spec)
    assert plan.nodes[0].tool == "format_text"


def test_rules_route_connectors():
    assert [n.tool for n in plan_by_rules("查一下法条").nodes] == ["legal_search"]
    assert [n.tool for n in plan_by_rules("查工商信息").nodes] == ["company_search"]
    assert [n.tool for n in plan_by_rules("帮我规划路线").nodes] == ["map_route"]


def test_unknown_intent_falls_back_to_memory():
    tools = [n.tool for n in plan_by_rules("随便聊聊").nodes]
    assert tools == ["memory_retrieve", "format_text"]
