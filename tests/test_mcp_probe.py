from __future__ import annotations

import httpx

from doubao_agent.config import load_config
from doubao_agent.sources.mcp import probe_http_mcp


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_probe_true_on_initialize_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/mcp"
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
        )

    assert await probe_http_mcp("http://host.test/mcp", client=_client(handler)) is True


async def test_probe_false_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    assert await probe_http_mcp("http://host.test/mcp", client=_client(handler)) is False


async def test_probe_false_on_non_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json", headers={"content-type": "text/plain"})

    assert await probe_http_mcp("http://host.test/mcp", client=_client(handler)) is False


async def test_probe_false_on_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await probe_http_mcp("http://host.test/mcp", client=_client(handler)) is False


def test_config_mcp_auto_discover_env(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text("port: 8766\n", encoding="utf-8")

    assert load_config(cfg_path).mcp_auto_discover is True

    monkeypatch.setenv("AGENT_MCP_AUTO_DISCOVER", "0")
    assert load_config(cfg_path).mcp_auto_discover is False
