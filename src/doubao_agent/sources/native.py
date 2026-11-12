from __future__ import annotations

import time
from typing import Any

from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec
from ..utils import dumps, infer_risk, jsonable
from .bridge import HostBridge
from .browser import browser_specs
from .desktop import desktop_specs


class NativeSource:
    """Every Doubao native capability, discovered from the host manifest.

    Discovery comes first, so new host capabilities need no code change.
    If the host exposes no manifest, fall back to the built-in browser/desktop
    catalog so the agent still has eyes and hands.
    """

    name = "native"
    kind = SourceKind.NATIVE

    def __init__(self, bridge: HostBridge | None, fallback: bool = True) -> None:
        self.bridge = bridge
        self.fallback = fallback
        self._discovered: list[ToolSpec] = []
        self._mode = "unavailable"

    def _to_spec(self, capability: dict[str, Any]) -> ToolSpec:
        name = str(capability["name"])
        description = str(capability.get("description") or "")
        schema = (
            capability.get("input_schema")
            or capability.get("inputSchema")
            or capability.get("parameters")
            or {"type": "object", "properties": {}}
        )
        raw_risk = capability.get("risk")
        risk: Risk
        if raw_risk:
            try:
                risk = Risk(str(raw_risk).upper())
            except ValueError:
                risk = infer_risk(name, description)
        else:
            risk = infer_risk(name, description)
        return ToolSpec(
            name=name,
            description=description,
            input_schema=schema,
            kind=SourceKind.NATIVE,
            risk=risk,
            source=self.name,
            idempotent=risk is Risk.READ,
            tags=tuple(capability.get("tags") or ()),
        )

    async def list_tools(self) -> list[ToolSpec]:
        if self.bridge is None:
            self._mode = "unavailable"
            return []

        try:
            capabilities = await self.bridge.list_capabilities()
        except Exception:  # noqa: BLE001 - discovery is best-effort
            capabilities = []

        if capabilities:
            self._discovered = [self._to_spec(c) for c in capabilities]
            self._mode = f"discovered:{len(self._discovered)}"
            return list(self._discovered)

        self._discovered = []
        if not self.fallback:
            self._mode = "manifest empty"
            return []
        self._mode = "fallback catalog"
        return browser_specs(self.name) + desktop_specs(self.name)

    async def call_tool(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        if self.bridge is None:
            return ToolResult(call_id=call.call_id, ok=False, error="host bridge not configured")
        try:
            result = await self.bridge.invoke(call.name, dict(call.arguments))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        content = result if isinstance(result, str) else dumps(jsonable(result))
        return ToolResult(
            call_id=call.call_id,
            ok=True,
            content=content,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def health(self) -> tuple[bool, str]:
        if self.bridge is None:
            return False, "host bridge not configured"
        if self._mode == "unavailable":
            return True, "host bridge connected (manifest pending)"
        return True, f"{self._mode} @ host bridge"

    async def aclose(self) -> None:
        close = getattr(self.bridge, "aclose", None)
        if callable(close):
            await close()
