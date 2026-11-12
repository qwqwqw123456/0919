"""规划与执行的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanNode:
    id: str
    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    needs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "params": self.params, "needs": self.needs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanNode":
        return cls(
            id=str(data["id"]),
            tool=str(data["tool"]),
            params=dict(data.get("params") or {}),
            needs=[str(x) for x in (data.get("needs") or [])],
        )


@dataclass
class Plan:
    intent: str
    nodes: list[PlanNode] = field(default_factory=list)

    def node(self, node_id: str) -> PlanNode | None:
        for item in self.nodes:
            if item.id == node_id:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"intent": self.intent, "nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        return cls(
            intent=str(data.get("intent", "")),
            nodes=[PlanNode.from_dict(n) for n in data.get("nodes", [])],
        )
