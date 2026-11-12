from __future__ import annotations

import httpx

from conftest import FakeBridge

from doubao_agent.config import Config, MCPServerConfig, load_config
from doubao_agent.registry import CapabilityRegistry
from doubao_agent.sources.bridge import HttpHostBridge
from doubao_agent.sources.mcp import MCPSource
from doubao_agent.sources.native import NativeSource
from doubao_agent.types import ToolCall


def test_bad_yaml_falls_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("permission_mode: [unclosed\n", encoding="utf-8")

    # 坏配置不应抛出异常，而是回退到默认值继续启动。
    cfg = load_config(path)

    assert isinstance(cfg, Config)
    assert cfg.permission_mode == "guarded"
    assert cfg.max_steps == 12


async def test_mcp_startup_failure_keeps_registry_alive() -> None:
    registry = CapabilityRegistry()
    registry.add_source(
        MCPSource(
            MCPServerConfig(name="bad", transport="stdio", command="/nonexistent/cmd")
        )
    )

    # 单个 MCP 源启动失败不影响注册表刷新与其余渠道。
    await registry.refresh()
    assert registry.list_tools() == []
    assert "bad" in registry.errors

    health = {row["source"]: row for row in await registry.health()}
    assert health["bad"]["ok"] is False

    await registry.aclose()


async def test_bridge_timeout_surfaces_as_tool_error() -> None:
    class TimeoutBridge(FakeBridge):
        async def invoke(self, capability: str, payload: dict):
            raise httpx.TimeoutException("timed out while waiting for host")

    source = NativeSource(TimeoutBridge([{"name": "slow.op", "description": "慢操作"}]))
    result = await source.call_tool(ToolCall(name="slow.op", arguments={}, call_id="t1"))

    assert result.ok is False
    assert "TimeoutException" in (result.error or "")
    assert "timed out" in (result.error or "")


async def test_bridge_invalid_capabilities_response_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bad-json":
            return httpx.Response(200, text="<html>not json</html>")
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    for path in ("/capabilities", "/bad-json"):
        bridge = HttpHostBridge(
            "http://host.test",
            capabilities_path=path,
            client=httpx.AsyncClient(transport=transport),
        )
        try:
            # 非法响应（500 或非 JSON）应降级为空清单，而不是抛异常。
            assert await bridge.list_capabilities() == []
        finally:
            await bridge.aclose()

    # 清单为空时 native 源仍给出兜底目录，保证工具面不空。
    class BrokenBridge(FakeBridge):
        async def list_capabilities(self) -> list[dict]:
            raise ValueError("malformed manifest")

    source = NativeSource(BrokenBridge([]))
    names = {spec.name for spec in await source.list_tools()}
    assert {"browser.open", "desktop.screenshot"} <= names
