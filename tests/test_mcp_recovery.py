from __future__ import annotations

import asyncio
import os

import pytest

from doubao_agent.config import MCPServerConfig
from doubao_agent.sources.mcp import MCPSource
from doubao_agent.sources.base import SourceError


class FakeSession:
    async def list_tools(self):
        class R:
            tools: list = []

        return R()


def make_source(**kw) -> MCPSource:
    cfg = MCPServerConfig(name="demo", transport="stdio", command="fake", **kw)
    return MCPSource(cfg)


def test_stdio_env_is_minimal(monkeypatch):
    monkeypatch.setenv("HOST_API_KEY", "sk-should-not-leave")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    src = make_source(env={"DEMO_FLAG": "1"})
    env = src._stdio_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["DEMO_FLAG"] == "1"
    assert "HOST_API_KEY" not in env


def test_stdio_env_inherit_opt_in(monkeypatch):
    monkeypatch.setenv("HOST_API_KEY", "sk-explicit")
    src = make_source(inherit_env=True)
    assert src._stdio_env()["HOST_API_KEY"] == "sk-explicit"


def test_failure_is_not_permanent(monkeypatch):
    src = make_source(retry_cooldown=0.0)

    async def boom():
        raise RuntimeError("连不上")

    monkeypatch.setattr(src, "_precheck", _noop)
    monkeypatch.setattr(src, "_connect", boom)
    with pytest.raises(SourceError):
        asyncio.run(src._ensure())
    assert src._fail_count == 1
    assert "第 1 次" in (src._failed or "")

    async def ok():
        return FakeSession()

    monkeypatch.setattr(src, "_connect", ok)
    session = asyncio.run(src._ensure())
    assert isinstance(session, FakeSession)
    assert src._failed is None and src._fail_count == 0


def test_failure_respects_cooldown(monkeypatch):
    src = make_source(retry_cooldown=30.0)
    src._failed = "MCP demo 初始化失败（第 1 次）：x"
    src._failed_at = __import__("time").monotonic()
    with pytest.raises(SourceError):
        asyncio.run(src._ensure())
    assert src._fail_count == 0  # 冷却期内不发新请求


async def _noop():
    return None
