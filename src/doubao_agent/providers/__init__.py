from __future__ import annotations

from typing import Any, Protocol


class ModelProvider(Protocol):
    """Minimal chat-completion interface with tool calling."""

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return {"content": str | None, "tool_calls": [{"id","name","arguments"}]}"""
        ...

    async def aclose(self) -> None: ...


def build_provider(cfg: dict[str, Any], bridge: Any = None) -> ModelProvider:
    provider = (cfg or {}).get("provider", "mock")
    if provider in ("host", "doubao", "native"):
        from .host import HostProvider

        if bridge is None:
            # 未配置宿主桥时退回 mock，避免运行时崩溃；仅用于本地无宿主开发。
            from .mock import MockProvider

            return MockProvider(script=cfg.get("script") if cfg else None)
        return HostProvider(
            bridge=bridge,
            capability=cfg.get("capability", "model.complete"),
            model=cfg.get("model"),
            temperature=float(cfg.get("temperature", 0.2)),
        )
    if provider in ("openai", "openai-compatible", "openai_compat"):
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
            api_key=cfg.get("api_key", ""),
            api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=float(cfg.get("temperature", 0.2)),
            extra_headers=cfg.get("headers") or {},
        )
    from .mock import MockProvider

    return MockProvider(script=cfg.get("script") if cfg else None)
