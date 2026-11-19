"""阶段 0 止血的验收用例：状态码识别、降级可见、配置透传、预算按 run 独立、监听准入。"""

from __future__ import annotations

import httpx

from doubao_agent.cli import check_serve_host
from doubao_agent.core.agent import Agent, _retryable, _status_of
from doubao_agent.core.audit import AuditLog
from doubao_agent.core.memory import MemoryStore
from doubao_agent.core.permissions import PermissionPolicy
from doubao_agent.core.session import SessionStore
from doubao_agent.core.usage import UsageMeter


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - 必然触发
        return exc
    raise AssertionError("raise_for_status 应抛 HTTPStatusError")


def test_httpx_status_error_is_read_from_response() -> None:
    """真实 httpx 异常的状态码挂在 response 上，不能被误判成"无状态码可重试"。"""
    auth = _http_status_error(401)
    assert _status_of(auth) == 401
    assert _retryable(auth) is False

    forbidden = _http_status_error(403)
    assert _retryable(forbidden) is False

    overloaded = _http_status_error(503)
    assert _status_of(overloaded) == 503
    assert _retryable(overloaded) is True


class AuthFailProvider:
    """始终 401：密钥错不该重试，应直接进降级分支。"""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        raise _http_status_error(401)

    async def aclose(self) -> None:
        return None


class FixedProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        return {"content": self.content, "tool_calls": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    async def aclose(self) -> None:
        return None


def _agent(tmp_path, registry, provider, fallback=None) -> Agent:
    db = tmp_path / "agent.db"
    return Agent(
        registry=registry,
        provider=provider,
        policy=PermissionPolicy("auto"),
        sessions=SessionStore(db),
        audit=AuditLog(tmp_path / "runs"),
        memory=MemoryStore(db),
        max_steps=3,
        fallback_provider=fallback,
        provider_retries=2,
        retry_delay=0.0,
    )


async def test_degraded_answer_is_visible_and_survives_second_run(tmp_path) -> None:
    """降级要对用户可见；上一次降级残留不能把下一次 run 打崩。"""
    from conftest import FakeSource, _registry

    registry = await _registry(FakeSource())
    fallback = FixedProvider("备用答复")
    agent = _agent(tmp_path, registry, AuthFailProvider(), fallback=fallback)
    events: list[dict] = []

    first = await agent.run("第一次", on_event=events.append)

    assert first["status"] == "completed"
    assert first["degraded"].startswith("status=401")
    assert "降级答复" in first["answer"] and "备用答复" in first["answer"]
    assert any(event["type"] == "degraded" for event in events)

    events_log = AuditLog(tmp_path / "runs").read(first["session_id"])
    assert any(event["event"] == "degraded" for event in events_log)

    # 主端点恢复正常后，同一实例再跑一次：不应复读旧降级标记，也不应抛 UnboundLocalError。
    agent.provider = FixedProvider("正常答复")
    agent.last_degraded = "stale reason"
    second = await agent.run("第二次")

    assert second["status"] == "completed"
    assert "degraded" not in second
    assert second["answer"] == "正常答复"


async def test_budget_is_per_run_not_process_wide(tmp_path) -> None:
    """单次 run 只用自己的预算表；进程级计量只做累加，不参与预算判定。"""
    from conftest import FakeSource, _registry

    registry = await _registry(FakeSource())
    agent = _agent(tmp_path, registry, FixedProvider("答复"))
    agent.usage = UsageMeter(max_calls=1)
    agent.total_usage = UsageMeter()

    first = await agent.run("第一次")
    assert first["status"] == "completed"
    assert first["usage"]["calls"] == 1
    assert agent.usage.exceeded() is not None  # 本次预算已用尽

    # 运行时每次 run 换一张新预算表（Runtime._run_meter），第二次不会被上一次拖死。
    agent.usage = UsageMeter(max_calls=1)
    second = await agent.run("第二次")
    assert second["status"] == "completed"
    assert second["usage"]["calls"] == 1
    assert agent.total_usage.calls == 2
    assert agent.total_usage.exceeded() is None  # 汇总表不设预算


async def test_runtime_spawn_agent_carries_fallback_and_retries(tmp_path) -> None:
    """配置里写的备用模型/重试/摘要必须真的透传到每次运行的 Agent。"""
    from doubao_agent.config import Config
    from doubao_agent.runtime import Runtime

    cfg = Config(
        workspace=tmp_path,
        runs_dir=tmp_path / "runs",
        db_path=tmp_path / "agent.db",
        skills_dir=tmp_path / "skills",
        scheduler_enabled=False,
        planning_enabled=False,
        model={"provider": "mock"},
        provider_fallback="mock",
        provider_retries=2,
        provider_retry_delay=0.5,
        context_model_summary=True,
        max_calls=1,
    )
    runtime = await Runtime.build(cfg)
    try:
        spawned = runtime.spawn_agent()
        assert spawned.fallback_provider is not None
        assert spawned.provider_retries == 2
        assert spawned.retry_delay == 0.5
        assert spawned.model_summary is True
        # 每次 spawn 都是独立的运行预算表，进程级计量单独汇总。
        assert spawned.usage.max_calls == 1
        assert spawned.total_usage is runtime.usage

        first = await runtime.start_run("第一次")
        second = await runtime.start_run("第二次")
        assert first["status"] == "completed" and second["status"] == "completed"
        assert runtime.usage.calls == 2
    finally:
        await runtime.aclose()


def test_serve_host_requires_token_when_exposed() -> None:
    assert check_serve_host("127.0.0.1", "") is None
    assert check_serve_host("localhost", "") is None
    assert check_serve_host("::1", "") is None
    assert check_serve_host("0.0.0.0", "tok") is None
    assert check_serve_host("0.0.0.0", "") is not None
