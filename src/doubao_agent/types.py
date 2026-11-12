from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Risk(str, Enum):
    """Reversibility level. Permissions are gated by this, not by allowlists."""

    READ = "L0"          # 查询 / 读取 / 截图
    REVERSIBLE = "L1"    # 可撤销的写操作
    IRREVERSIBLE = "L2"  # 删除 / 支付 / 对外发布
    SENIOR = "L3"        # 系统级、批量覆盖


class SourceKind(str, Enum):
    MCP = "mcp"
    API = "api"
    BROWSER = "browser"
    DESKTOP = "desktop"
    NATIVE = "native"
    BUILTIN = "builtin"
    SKILL = "skill"
    KNOWLEDGE = "knowledge"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    kind: SourceKind = SourceKind.BUILTIN
    risk: Risk = Risk.READ
    tags: tuple[str, ...] = ()
    idempotent: bool = True
    source: str = "builtin"

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass
class ToolResult:
    call_id: str
    ok: bool
    content: Any = None
    error: str | None = None
    elapsed_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_text(self, limit: int = 6000) -> str:
        if not self.ok:
            return f"[ERROR] {self.error}"
        text = self.content if isinstance(self.content, str) else repr(self.content)
        return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


@dataclass
class Message:
    role: str                      # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data
