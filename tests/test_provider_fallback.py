from __future__ import annotations

import pytest

from doubao_agent.core.agent import Agent, _retryable, _status_of
from doubao_agent.core.memory import MemoryStore


class EndpointError(Exception):
    def __init__(self, status_code: int, message: str = "endpoint failed") -> None:
        super().__init__(message)
        self.status_code = status_code


class ScriptedProvider:
    """按次数抛错的假 provider，用来验证重试与降级分支。"""

    def __init__(self, fails: int, exc: Exception | None = None, content: str = "ok") -> None:
        self.fails = fails
        self.exc = exc or TimeoutError("read timeout")
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return {"content": self.content, "tool_calls": []}

    async def aclose(self) -> None:
        return None


def _agent(tmp_path, provider, *, fallback=None, retries: int = 1) -> Agent:
    return Agent(
        registry=object(),
        provider=provider,
        policy=object(),
        sessions=object(),
        audit=object(),
        memory=MemoryStore(tmp_path / "agent.db"),
        fallback_provider=fallback,
        provider_retries=retries,
        retry_delay=0.0,
    )


def test_status_and_retryable_classification() -> None:
    assert _status_of(EndpointError(401)) == 401
    assert _status_of(TimeoutError("x")) is None
    assert _retryable(EndpointError(503)) is True
    assert _retryable(EndpointError(429)) is True
    assert _retryable(EndpointError(401)) is False
    assert _retryable(EndpointError(403)) is False
    assert _retryable(TimeoutError("read timeout")) is True


async def test_retry_then_success(tmp_path) -> None:
    provider = ScriptedProvider(fails=1)
    agent = _agent(tmp_path, provider, retries=2)

    reply = await agent._complete([{"role": "user", "content": "hi"}], [])

    assert reply["content"] == "ok"
    assert provider.calls == 2
    assert agent.last_degraded is None
    agent.memory.close()


async def test_retry_exhausted_without_fallback_raises(tmp_path) -> None:
    provider = ScriptedProvider(fails=10)
    agent = _agent(tmp_path, provider, retries=1)

    with pytest.raises(TimeoutError):
        await agent._complete([{"role": "user", "content": "hi"}], [])

    assert provider.calls == 2  # 1 次 + 1 次重试
    agent.memory.close()


async def test_auth_error_skips_retry_and_degrades(tmp_path) -> None:
    provider = ScriptedProvider(fails=10, exc=EndpointError(401))
    fallback = ScriptedProvider(fails=0, content="备用答复")
    agent = _agent(tmp_path, provider, fallback=fallback, retries=3)

    reply = await agent._complete([{"role": "user", "content": "hi"}], [])

    assert provider.calls == 1, "401 不该重试"
    assert fallback.calls == 1
    assert reply["content"] == "备用答复"
    assert "401" in reply["degraded"]
    assert reply["degraded_provider"] == "ScriptedProvider"
    assert agent.last_degraded
    agent.memory.close()


async def test_both_providers_down_raises_with_reason(tmp_path) -> None:
    provider = ScriptedProvider(fails=10, exc=EndpointError(503))
    fallback = ScriptedProvider(fails=10, exc=EndpointError(500))
    agent = _agent(tmp_path, provider, fallback=fallback, retries=1)

    with pytest.raises(RuntimeError) as excinfo:
        await agent._complete([{"role": "user", "content": "hi"}], [])

    assert "主备模型均不可用" in str(excinfo.value)
    agent.memory.close()
