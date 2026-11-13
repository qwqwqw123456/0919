from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "workspace": str(tmp_path),
                "runs_dir": str(tmp_path / "runs"),
                "db_path": str(tmp_path / "agent.db"),
                "skills_dir": str(tmp_path / "skills"),
                "permission_mode": "guarded",
                "script_risk": "L2",
                "mcp_auto_discover": False,
                "evolution": {"enabled": False},
                "model": {"provider": "mock"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENT_") and key != "DOUBAO_AGENT_CONFIG"
    }
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _params(config_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "doubao_agent.cli", "--config", str(config_path), "mcp-serve"],
        env=_env(),
    )


def _text(result) -> str:
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", "") == "text"
    )


@pytest.mark.asyncio
async def test_mcp_server_lists_and_calls_readonly_tool(tmp_path):
    async with stdio_client(_params(_config(tmp_path))) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            assert info.serverInfo.name == "doubao-agent"

            listed = await session.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert "system__status" in tools
            assert "script__run_shell" in tools
            assert tools["system__status"].inputSchema["type"] == "object"

            result = await session.call_tool("system__status", {})
            assert result.isError is False
            assert "平台" in _text(result)


@pytest.mark.asyncio
async def test_mcp_server_denies_gated_tool_without_approver(tmp_path):
    async with stdio_client(_params(_config(tmp_path))) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("script__run_shell", {"command": "echo hi"})
            assert result.isError is True
            body = _text(result)
            assert "denied" in body or "未配置审批人" in body


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_http_transport_refuses_non_loopback_without_token(tmp_path):
    env = _env()
    env.pop("DOUBAO_MCP_TOKEN", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "doubao_agent.cli",
            "--config",
            str(_config(tmp_path)),
            "mcp-serve",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            str(_free_port()),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 2
    assert "必须配置 --token" in result.stderr


async def _wait_http(url: str, process: asyncio.subprocess.Process, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last: Exception | None = None
    async with httpx.AsyncClient() as client:
        while loop.time() < deadline:
            if process.returncode is not None:
                stderr = (await process.stderr.read()).decode(errors="replace")
                raise AssertionError(f"server exited early: {stderr[-2000:]}")
            try:
                response = await client.get(url, timeout=1.0)
                if response.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001 - retry until the listener is ready
                last = exc
            await asyncio.sleep(0.1)
    raise AssertionError(f"server did not start: {last}")


@pytest.mark.asyncio
async def test_streamable_http_transport_lists_and_calls_tools(tmp_path):
    streamable_http = pytest.importorskip("mcp.client.streamable_http")
    port = _free_port()
    token = "test-token"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "doubao_agent.cli",
        "--config",
        str(_config(tmp_path)),
        "mcp-serve",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--token",
        token,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_env(),
    )
    try:
        base = f"http://127.0.0.1:{port}"
        await _wait_http(f"{base}/healthz", process)

        async with httpx.AsyncClient() as client:
            denied = await client.post(
                f"{base}/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
            assert denied.status_code == 401
            assert denied.headers.get("www-authenticate") == "Bearer"

        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}
        ) as http_client:
            async with streamable_http.streamable_http_client(
                f"{base}/mcp", http_client=http_client
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    info = await session.initialize()
                    assert info.serverInfo.name == "doubao-agent"

                    listed = await session.list_tools()
                    tools = {tool.name for tool in listed.tools}
                    assert "system__status" in tools
                    assert "script__run_shell" in tools

                    result = await session.call_tool("system__status", {})
                    assert result.isError is False
                    assert "平台" in _text(result)
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        else:
            stderr = (await process.stderr.read()).decode(errors="replace")
            if process.returncode != 0:
                print(stderr[-2000:], file=sys.stderr)
