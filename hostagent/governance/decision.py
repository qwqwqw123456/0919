"""治理层：用规则预授权接管宿主 HITL。

宿主契约要求「人工批准」的节点（send_message / create_calendar / send_file），
在这里被改写为：命中预授权白名单即自动放行，未命中则拒绝（不打断、不等待人工）。
可逆/只读节点一律自动放行。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..contract import Approval, Node, Risk
from .ledger import Ledger
from .policy import PolicyStore

AUTO = "auto_approved"
PREAUTH = "preauthorized"
DENIED = "denied"


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str
    approval_id: str | None = None
    operator: str | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict in (AUTO, PREAUTH)


class Gate:
    """模式：auto = 只读/本地/文档自动，写与外发需预授权；strict = 仅只读/本地自动。"""

    def __init__(self, policy: PolicyStore, ledger: Ledger | None = None, mode: str = "auto") -> None:
        if mode not in ("auto", "strict"):
            raise ValueError(f"未知治理模式: {mode}")
        self.policy = policy
        self.ledger = ledger
        self.mode = mode

    def decide(self, node: Node, params: dict) -> Decision:
        decision = self._decide(node, params)
        if self.ledger is not None:
            self.ledger.record(
                "decision",
                tool=node.tool,
                action=node.action,
                risk=node.risk.value,
                contract_approval=node.approval.value,
                verdict=decision.verdict,
                reason=decision.reason,
                approval_id=decision.approval_id,
                operator=decision.operator,
            )
        return decision

    def _decide(self, node: Node, params: dict) -> Decision:
        auto_risks = {Risk.READ, Risk.LOCAL}
        if self.mode == "auto":
            auto_risks.add(Risk.DOC)
        if node.risk in auto_risks and node.approval is Approval.NONE:
            return Decision(AUTO, f"{node.risk.value} 自动放行")

        rule = self.policy.match_rule(node.tool, params)
        if rule is not None:
            return Decision(
                PREAUTH,
                f"命中预授权规则 #{rule['id']} ({rule['dimension']}={rule['value']})",
                approval_id=f"pa-{uuid.uuid4().hex[:8]}",
                operator=rule.get("operator") or "policy",
            )

        return Decision(
            DENIED,
            f"{node.risk.value}/{node.approval.value}：未命中预授权白名单，已拒绝（不等待人工）",
        )
