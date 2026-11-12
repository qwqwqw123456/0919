from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from ..types import Risk, ToolCall, ToolSpec

Approver = Callable[[ToolSpec, ToolCall, str], Awaitable[bool]]


class PermissionMode(str, Enum):
    AUTO = "auto"          # 除 L3 外全自动
    GUARDED = "guarded"    # L2 起确认
    MANUAL = "manual"      # 除只读外都确认


@dataclass
class Decision:
    allowed: bool
    needs_approval: bool
    reason: str = ""


class PermissionPolicy:
    """Reversibility-based gating instead of an allowlist."""

    def __init__(self, mode: str | PermissionMode = PermissionMode.GUARDED,
                 approver: Approver | None = None) -> None:
        self.mode = PermissionMode(mode)
        self.approver = approver

    async def authorize(self, spec: ToolSpec, call: ToolCall) -> Decision:
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
