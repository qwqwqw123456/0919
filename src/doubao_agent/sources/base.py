from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import ToolCall, ToolResult, ToolSpec


@runtime_checkable
class CapabilitySource(Protocol):
    """Every channel (MCP / API / browser / desktop) implements this protocol.

    The upper layers never branch on the channel kind: they only see ToolSpec.
    """

    name: str

    async def list_tools(self) -> list[ToolSpec]: ...

    async def call_tool(self, call: ToolCall) -> ToolResult: ...

    async def health(self) -> tuple[bool, str]: ...

    async def aclose(self) -> None: ...


class SourceError(RuntimeError):
    pass
