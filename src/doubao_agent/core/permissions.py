from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any, Awaitable, Callable

from ..types import Risk, ToolCall, ToolSpec

Approver = Callable[[ToolSpec, ToolCall, str], Awaitable[bool]]

_RISK_RANK = {
    Risk.READ: 0,
    Risk.REVERSIBLE: 1,
    Risk.IRREVERSIBLE: 2,
    Risk.SENIOR: 3,
}


class PermissionMode(str, Enum):
    AUTO = "auto"          # 除 L3 外全自动
    GUARDED = "guarded"    # L2 起确认
    MANUAL = "manual"      # 除只读外都确认


@dataclass
class Decision:
    allowed: bool
    needs_approval: bool
    reason: str = ""


@dataclass(frozen=True)
class PolicyRule:
    """规则预授权：命中即放行，替代无人值守场景下的人工审批。

    tool 支持通配符（如 `send_*`）；risk 是允许的最高风险等级；
    arguments 是必须全部相等的参数约束（如 {"target_id": "self"}）。
    """

    tool: str = "*"
    risk: str = Risk.SENIOR.value
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = "预授权规则"

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "PolicyRule":
        return cls(
            tool=str(item.get("tool") or "*"),
            risk=str(item.get("risk") or Risk.SENIOR.value).upper(),
            arguments=dict(item.get("arguments") or {}),
            reason=str(item.get("reason") or "预授权规则"),
        )

    def allows(self, spec: ToolSpec, call: ToolCall) -> bool:
        # 注册表会把工具名限定为 `<source>.<name>`，规则按裸名或全名都应能命中。
        names = {spec.name, spec.name.rsplit(".", 1)[-1]}
        if not any(fnmatchcase(name, self.tool) for name in names):
            return False
        try:
            ceiling = _RISK_RANK[Risk(self.risk)]
        except (KeyError, ValueError):
            return False
        if _RISK_RANK[spec.risk] > ceiling:
            return False
        return all(call.arguments.get(key) == value for key, value in self.arguments.items())


class PermissionPolicy:
    """Reversibility-based gating plus explicit preauthorization rules."""

    def __init__(
        self,
        mode: str | PermissionMode = PermissionMode.GUARDED,
        approver: Approver | None = None,
        rules: list[dict[str, Any] | PolicyRule] | None = None,
    ) -> None:
        try:
            self.mode = PermissionMode(mode)
        except ValueError:
            self.mode = PermissionMode.GUARDED
        self.approver = approver
        self.rules = [
            rule if isinstance(rule, PolicyRule) else PolicyRule.from_dict(rule)
            for rule in rules or []
        ]

    def _match_rule(self, spec: ToolSpec, call: ToolCall) -> PolicyRule | None:
        return next((rule for rule in self.rules if rule.allows(spec, call)), None)

    async def authorize(self, spec: ToolSpec, call: ToolCall) -> Decision:
        rule = self._match_rule(spec, call)
        if rule is not None:
            return Decision(True, False, f"policy: {rule.reason}")

        risk = spec.risk

        if self.mode is PermissionMode.AUTO:
            if risk is Risk.SENIOR:
                return await self._ask(spec, call, "L3 高危操作")
            return Decision(True, False, "auto")

        if self.mode is PermissionMode.GUARDED:
            if risk in (Risk.IRREVERSIBLE, Risk.SENIOR):
                return await self._ask(spec, call, f"{risk.value} 不可逆操作")
            return Decision(True, False, "guarded")

        # MANUAL
        if risk is Risk.READ:
            return Decision(True, False, "manual: read-only")
        return await self._ask(spec, call, f"{risk.value} 写操作")

    async def _ask(self, spec: ToolSpec, call: ToolCall, reason: str) -> Decision:
        if self.approver is None:
            return Decision(False, True, f"{reason}：未配置审批人，已拒绝")
        approved = await self.approver(spec, call, reason)
        return Decision(approved, True, f"{reason}：{'已批准' if approved else '已拒绝'}")
