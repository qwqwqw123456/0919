from __future__ import annotations

import httpx

from doubao_agent.cli import _serve_binding
from doubao_agent.config import Config
from doubao_agent.core.agent import Agent, _retryable, _status_of
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.session import SessionStore
from doubao_agent.core.usage import UsageMeter
from doubao_agent.registry import CapabilityRegistry
from doubao_agent.runtime import Runtime
from doubao_agent.types import Risk, ToolCall, ToolSpec


class FailingProvider:
    async def complete(self, messages, tools):
        raise TimeoutError("read timeout")

    async def aclose(self) -> None:
        return None


class FallbackProvider:
    def __init__(self, content: str = "备用答复") -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        return {"content": self.content, "tool_calls": []}

    async def aclose(self) -> None:
        return None


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("endpoint failed", request=request, response=response)


def test_serve_defaults_to_loopback_and_rejects_open_binding() -> None:
    host, error = _serve_binding("127.0.0.1", "")
    assert host == "127.0.0.1"
    assert error is None

    host, error = _serve_binding("0.0.0.0", "")
    assert host == "0.0.0.0"
    assert error and "auth_token" in error

    host, error = _serve_binding("0.0.0.0", "s3cret")
    assert host == "0.0.0.0"
    assert error is None


def test_httpx_status_errors_are_not_retried() -> None:
    for status in (400, 401, 403, 404):
        exc = _http_error(status)
        assert _status_of(exc) == status
        assert _retryable(exc) is False

    for status in (429, 500, 503):
        assert _retryable(_http_error(status)) is True


async def test_script_risk_defaults_to_confirmation_and_escalates_danger() -> None:
    assert Config().script_risk == "L2"
    spec = ToolSpec(
        name="script.run_shell",
        description="run shell",
        risk=Risk.IRREVERSIBLE,
        source="script",
    )

    guarded = PermissionPolicy("guarded")
    benign = await guarded.authorize(
        spec, ToolCall(name="script.run_shell", arguments={"command": "ls -la"})
    )
    assert benign.needs_approval is True
    assert benign.allowed is False

    auto = PermissionPolicy("auto")
    allowed = await auto.authorize(
        spec, ToolCall(name="script.run_shell", arguments={"command": "ls -la"})
    )
    assert allowed.allowed is True

    danger = await auto.authorize(
        spec, ToolCall(name="script.run_shell", arguments={"command": "rm -rf /tmp/x"})
    )
    assert danger.needs_approval is True
    assert danger.allowed is False


def test_usage_child_meters_are_isolated_and_parent_accumulates() -> None:
    parent = UsageMeter(max_calls=2)
    first = parent.child()
    second = parent.child()

    first.record({"prompt_tokens": 3, "completion_tokens": 2})
    first.record(None)

    assert first.calls == 2
    assert first.exceeded() is not None
    assert second.calls == 0
    assert second.exceeded() is None
    assert parent.calls == 2
    assert parent.prompt_tokens == 3
    assert parent.completion_tokens == 2


async def _runtime(tmp_path, **overrides) -> Runtime:
    cfg = Config(
        workspace=tmp_path,
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "agent.db",
        skills_dir=tmp_path / "skills",
        scheduler_enabled=False,
        mcp_auto_discover=False,
        model={"provider": "mock"},
        **overrides,
    )
    cfg.ensure_dirs()
    return await Runtime.build(cfg)


async def test_spawn_agent_inherits_provider_resilience(tmp_path) -> None:
    runtime = await _runtime(
        tmp_path,
        planning_enabled=True,
        provider_retries=3,
        provider_fallback="mock",
        context_model_summary=True,
    )
    try:
        agent = runtime.spawn_agent()
        assert agent.fallback_provider is not None
        assert agent.provider_retries == 3
        assert agent.model_summary is True
        assert agent.usage is not runtime.usage
        assert agent.usage.parent is runtime.usage
        assert agent.planner is not None
        assert agent.planner.usage is agent.usage
    finally:
        await runtime.aclose()


async def test_each_run_gets_a_fresh_budget(tmp_path) -> None:
    runtime = await _runtime(tmp_path, max_calls=1, planning_enabled=False)
    try:
        first = await runtime.start_run("第一轮")
        second = await runtime.start_run("第二轮")

        assert first["status"] == "completed"
        assert second["status"] == "completed"
        assert first["usage"]["calls"] == 1
        assert second["usage"]["calls"] == 1
        assert runtime.usage.calls == 2
    finally:
        await runtime.aclose()


async def test_degraded_run_is_visible_and_survives_repeat(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "agent.db")
    sessions = SessionStore(tmp_path / "agent.db")
    audit = AuditLog(tmp_path / "runs")
    fallback = FallbackProvider()
    agent = Agent(
        registry=CapabilityRegistry(),
        provider=FailingProvider(),
        policy=PermissionPolicy("auto"),
        sessions=sessions,
        audit=audit,
        memory=memory,
        fallback_provider=fallback,
        provider_retries=0,
    )
    try:
        first = await agent.run("第一次")
        assert first["status"] == "completed"
        assert first["degraded"]
        assert "主模型端点不可用" in first["answer"]
        assert "备用答复" in first["answer"]
        events = [record["event"] for record in audit.read(first["session_id"])]
        assert "degraded" in events

        second = await agent.run("第二次")
        assert second["status"] == "completed"
        assert fallback.calls == 2
    finally:
        memory.close()
        sessions.close()
