from __future__ import annotations

from dataclasses import replace
from typing import Any

from .sources.base import CapabilitySource
from .types import ToolCall, ToolResult, ToolSpec
from .utils import dumps


class CapabilityRegistry:
    """Single entry point for every capability, regardless of its channel."""

    def __init__(self) -> None:
        self._sources: dict[str, CapabilitySource] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._dispatch: dict[str, tuple[CapabilitySource, str]] = {}
        self._errors: dict[str, str] = {}

    def add_source(self, source: CapabilitySource) -> None:
        self._sources[source.name] = source

    def has_source(self, name: str) -> bool:
        return name in self._sources

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    async def refresh(self) -> None:
        specs: dict[str, ToolSpec] = {}
        dispatch: dict[str, tuple[CapabilitySource, str]] = {}
        errors: dict[str, str] = {}

        for source in self._sources.values():
            try:
                tools = await source.list_tools()
            except Exception as exc:  # noqa: BLE001 - one bad channel must not kill the rest
                errors[source.name] = f"{type(exc).__name__}: {exc}"
                continue
            for spec in tools:
                # A namespaced name (e.g. browser.open, demo.health) is kept as-is;
                # bare names get the source prefix (e.g. echo.add).
                qualified = spec.name if "." in spec.name else f"{source.name}.{spec.name}"
                specs[qualified] = replace(spec, name=qualified, source=source.name)
                dispatch[qualified] = (source, spec.name)

        self._specs = specs
        self._dispatch = dispatch
        self._errors = errors

    def list_tools(self) -> list[ToolSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def get(self, name: str) -> ToolSpec | None:
        key = self.resolve(name)
        return self._specs.get(key) if key else None

    def resolve(self, name: str) -> str | None:
        if name in self._specs:
            return name
        suffix = "." + name
        matches = [k for k in self._specs if k.endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    async def call(self, call: ToolCall) -> ToolResult:
        key = self.resolve(call.name)
        if key is None:
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown tool: {call.name}")
        source, original = self._dispatch[key]
        return await source.call_tool(
            ToolCall(name=original, arguments=call.arguments, call_id=call.call_id)
        )

    async def cancel(self) -> None:
        """Best-effort cancellation across all sources (e.g. kill running scripts)."""
        for source in self._sources.values():
            cancel = getattr(source, "cancel", None)
            if callable(cancel):
                try:
                    await cancel()
                except Exception:  # noqa: BLE001 - best-effort
                    pass

    async def health(self) -> list[dict[str, Any]]:
        rows = []
        for name, source in self._sources.items():
            try:
                ok, detail = await source.health()
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            rows.append({"source": name, "ok": ok, "detail": detail})
        return rows

    def snapshot(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for spec in self._specs.values():
            by_kind[spec.kind.value] = by_kind.get(spec.kind.value, 0) + 1
        return {
            "sources": list(self._sources),
            "tools": len(self._specs),
            "by_kind": by_kind,
            "errors": self._errors,
        }

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai() for spec in self.list_tools()]

    async def aclose(self) -> None:
        for source in self._sources.values():
            try:
                await source.aclose()
            except Exception:  # noqa: BLE001 - best-effort shutdown
                pass

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return dumps(self.snapshot())
