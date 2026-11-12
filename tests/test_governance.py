from __future__ import annotations

from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.types import Risk, ToolCall, ToolSpec


async def test_guarded_mode_denies_irreversible_without_approver() -> None:
    policy = PermissionPolicy("guarded")
    spec = ToolSpec(name="crm.delete", description="delete a record", risk=Risk.IRREVERSIBLE)
    decision = await policy.authorize(spec, ToolCall(name="crm.delete"))
    assert decision.needs_approval is True
    assert decision.allowed is False


async def test_guarded_mode_allows_read() -> None:
    policy = PermissionPolicy("guarded")
    spec = ToolSpec(name="crm.list", description="list records", risk=Risk.READ)
    decision = await policy.authorize(spec, ToolCall(name="crm.list"))
    assert decision.allowed is True


async def test_policy_rule_preauthorizes_without_approver() -> None:
    policy = PermissionPolicy(
        "guarded",
        rules=[
            {
                "tool": "send_message",
                "arguments": {"target_id": "self"},
                "risk": "L2",
                "reason": "仅本人",
            }
        ],
    )
    spec = ToolSpec(name="send_message", description="send", risk=Risk.IRREVERSIBLE)

    hit = await policy.authorize(
        spec, ToolCall(name="send_message", arguments={"target_id": "self"})
    )
    assert hit.allowed is True
    assert hit.needs_approval is False
    assert "仅本人" in hit.reason

    miss = await policy.authorize(
        spec, ToolCall(name="send_message", arguments={"target_id": "other"})
    )
    assert miss.allowed is False


async def test_policy_rule_respects_risk_ceiling() -> None:
    policy = PermissionPolicy("guarded", rules=[{"tool": "*", "risk": "L1"}])
    spec = ToolSpec(name="crm.delete", description="delete", risk=Risk.SENIOR)
    decision = await policy.authorize(spec, ToolCall(name="crm.delete"))
    assert decision.allowed is False


async def test_policy_rule_matches_source_prefixed_tool_name() -> None:
    policy = PermissionPolicy(
        "guarded",
        rules=[{"tool": "send_message", "arguments": {"target_id": "self"}, "risk": "L2"}],
    )
    spec = ToolSpec(name="native.send_message", description="send", risk=Risk.IRREVERSIBLE)
    decision = await policy.authorize(
        spec, ToolCall(name="native.send_message", arguments={"target_id": "self"})
    )
    assert decision.allowed is True
    assert decision.needs_approval is False
