from __future__ import annotations

import os
import sys
from pathlib import Path

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
