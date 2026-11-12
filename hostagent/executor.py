"""DAG 执行器：按契约顺序执行计划，逐节点过治理门。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import subagents
from .contract import get_node
from .gateway import DelegateUnavailable, HostRuntimeGateway
from .governance import Gate, Ledger
from .memory import MemoryStore
from .models import Plan, PlanNode
from .planner import validate

_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.\-]+)\}")


@dataclass
class NodeOutcome:
    node_id: str
    tool: str
    status: str                    # completed / failed / denied / blocked / unavailable
    output: Any = None
    error: str = ""
    verdict: str = ""
    reason: str = ""
    approval_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "tool": self.tool,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "verdict": self.verdict,
            "reason": self.reason,
            "approval_id": self.approval_id,
        }


def resolve(value: Any, ctx: dict[str, Any], intent: str) -> Any:
    """解析 {intent} / {node_id} / {node_id.key} / {provides} 占位符。"""
    if isinstance(value, str):
        exact = _PLACEHOLDER.fullmatch(value)
        if exact:
            return _lookup(exact.group(1), ctx, intent, value)
        return _PLACEHOLDER.sub(lambda m: _as_text(_lookup(m.group(1), ctx, intent, m.group(0))), value)
    if isinstance(value, list):
        return [resolve(item, ctx, intent) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item, ctx, intent) for key, item in value.items()}
    return value


def _lookup(key: str, ctx: dict[str, Any], intent: str, fallback: Any) -> Any:
    if key == "intent":
        return intent
    if key in ctx:
        return ctx[key]
    base, _, sub = key.partition(".")
    if base in ctx:
        value = ctx[base]
        if isinstance(value, dict) and sub in value:
            return value[sub]
    return fallback


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class Executor:
    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        gateway: HostRuntimeGateway | None = None,
        gate: Gate | None = None,
        ledger: Ledger | None = None,
        output_dir: str | Path = "output",
        dry_run: bool = False,
        fetch: Callable[[str], str] | None = None,
        lark: Callable[[list[str]], tuple[int, str, str]] | None = None,
    ) -> None:
        self.memory = memory
        self.gateway = gateway or HostRuntimeGateway()
        self.gate = gate
        self.ledger = ledger
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.fetch = fetch
        self.lark = lark

    def run(self, plan: Plan) -> dict[str, Any]:
        validate(plan)
        if self.ledger:
            self.ledger.record("plan", intent=plan.intent, nodes=[n.to_dict() for n in plan.nodes])

        ctx: dict[str, Any] = {}
        outcomes: dict[str, NodeOutcome] = {}
        pending = list(plan.nodes)

        while pending:
            progressed = False
            for node in list(pending):
                if any(dep not in outcomes for dep in node.needs):
                    continue
                failed = [d for d in node.needs if outcomes[d].status != "completed"]
                if failed:
                    outcomes[node.id] = NodeOutcome(
                        node.id, node.tool, "blocked", error=f"上游未成功: {', '.join(failed)}"
                    )
                    pending.remove(node)
                    progressed = True
                    continue
                outcomes[node.id] = self._run_node(plan, node, ctx)
                pending.remove(node)
                progressed = True
            if not progressed:
                for node in pending:
                    outcomes[node.id] = NodeOutcome(node.id, node.tool, "blocked", error="依赖无法满足（疑似环）")
                break

        statuses = [o.status for o in outcomes.values()]
        completed = statuses.count("completed")
        if completed == len(statuses) and statuses:
            status = "completed"
        elif completed:
            status = "partial"
        else:
            status = "failed"

        artifacts = [
            o.output for o in outcomes.values()
            if isinstance(o.output, str) and o.output.endswith((".xlsx", ".pptx"))
        ]
        final_text = ctx.get("summary") or ctx.get("pick") or ""
        report = {
            "intent": plan.intent,
            "status": status,
            "total": len(statuses),
            "completed": completed,
            "final_text": final_text if isinstance(final_text, str) else _as_text(final_text),
            "artifacts": artifacts,
            "nodes": {node_id: outcome.to_dict() for node_id, outcome in outcomes.items()},
        }
        if self.ledger:
            self.ledger.record("finish", intent=plan.intent, status=status, completed=completed, total=len(statuses))
        return report

    def _run_node(self, plan: Plan, node: PlanNode, ctx: dict[str, Any]) -> NodeOutcome:
        spec = get_node(node.tool)
        assert spec is not None  # validate 已保证
        params = resolve(node.params, ctx, plan.intent)

        verdict = "auto_approved"
        reason = "未配置治理门"
        approval_id = None
        if self.gate is not None:
            decision = self.gate.decide(spec, params)
            verdict, reason, approval_id = decision.verdict, decision.reason, decision.approval_id
            if not decision.allowed:
                outcome = NodeOutcome(node.id, node.tool, "denied", verdict=verdict, reason=reason)
                self._log_node(outcome)
                return outcome

        try:
            if spec.delegate == "data_agent":
                result = subagents.data.do_run(
                    spec.action, params, memory=self.memory, fetch=self.fetch
                )
            elif spec.delegate == "delivery_agent":
                result = subagents.delivery.do_run(
                    spec.action,
                    params,
                    output_dir=self.output_dir,
                    dry_run=self.dry_run,
                    lark=self.lark,
                )
            elif spec.delegate == "external_runtime":
                output = self.gateway.invoke(spec.action, params)
                result = {"ok": True, "output": output}
            else:
                result = {"ok": False, "error": f"未知 delegate: {spec.delegate}"}
        except DelegateUnavailable as exc:
            outcome = NodeOutcome(node.id, node.tool, "unavailable", error=str(exc),
                                  verdict=verdict, reason=reason, approval_id=approval_id)
            self._log_node(outcome)
            return outcome
        except Exception as exc:
            outcome = NodeOutcome(node.id, node.tool, "failed", error=f"{type(exc).__name__}: {exc}",
                                  verdict=verdict, reason=reason, approval_id=approval_id)
            self._log_node(outcome)
            return outcome

        if not result.get("ok"):
            outcome = NodeOutcome(node.id, node.tool, "failed", error=str(result.get("error") or "未知失败"),
                                  verdict=verdict, reason=reason, approval_id=approval_id)
            self._log_node(outcome)
            return outcome

        output = result.get("output")
        ctx[node.id] = output
        if spec.produces:
            ctx[spec.produces] = output
        outcome = NodeOutcome(node.id, node.tool, "completed", output=output,
                              verdict=verdict, reason=reason, approval_id=approval_id)
        self._log_node(outcome)
        return outcome

    def _log_node(self, outcome: NodeOutcome) -> None:
        if self.ledger:
            self.ledger.record(
                "node",
                node=outcome.node_id,
                tool=outcome.tool,
                status=outcome.status,
                verdict=outcome.verdict,
                error=outcome.error,
                approval_id=outcome.approval_id,
            )
