"""策略层单测：默认值、覆盖优先级、热改、持久化只动 policy 段、Provider/Agent/A2A 真吃策略。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from doubao_agent.core.a2a_client import A2AClient
from dataclasses import fields

from doubao_agent.core.netpolicy import NetPolicy, ModelPolicy, PRESETS, Policy, PolicyError
from doubao_agent.providers.openai_compat import OpenAICompatProvider


def _yaml(tmp_path, body: str):
    p = tmp_path / "agent.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_defaults_and_families():
    perf = Policy(None, environ={})
    snap = perf.snapshot()
    assert set(snap) == {"net", "model", "preset", "version"}
    assert snap["net"].retries >= 0 and snap["model"].timeout > 0
    rows = perf.explain()
    assert len(rows) == len(fields(NetPolicy)) + len(fields(ModelPolicy))
    assert all(r["source"] == "default" for r in rows)


def test_yaml_overrides_and_preset(tmp_path):
    p = _yaml(tmp_path, "policy:\n  preset: turbo\n  model:\n    retries: 4\n")
    perf = Policy(p, environ={})
    assert perf.model.retries == 4          # yaml 盖 preset
    assert perf.net.keepalive == PRESETS["turbo"]["net"]["keepalive"]


def test_env_wins(tmp_path):
    p = _yaml(tmp_path, "policy:\n  model:\n    timeout: 60\n")
    perf = Policy(p, environ={"AGENT_POLICY_MODEL_TIMEOUT": "15"})
    assert perf.model.timeout == 15.0
    assert perf.explain()[0]["key"].startswith("net.")


def test_bad_values_rejected(tmp_path):
    p = _yaml(tmp_path, "policy:\n  model:\n    retries: -1\n")
    with pytest.raises(PolicyError):
        Policy(p, environ={})
    perf = Policy(None, environ={})
    with pytest.raises(PolicyError):
        perf.set("model.nope", 1)
    with pytest.raises(PolicyError):
        perf.preset("unknown")


def test_set_persists_only_policy_and_other_keys_survive(tmp_path):
    p = _yaml(tmp_path, "workspace: /tmp/x\npolicy:\n  preset: balanced\nmodel:\n  provider: mock\n")
    perf = Policy(p, environ={})
    perf.set("model.retries", 3)
    perf.set("net.read_timeout", 12)
    import yaml

    data = yaml.safe_load(p.read_text("utf-8"))
    assert data["workspace"] == "/tmp/x"
    assert data["model"]["provider"] == "mock"
    assert data["policy"]["model"]["retries"] == 3
    assert data["policy"]["net"]["read_timeout"] == 12
    assert Policy(p, environ={}).model.retries == 3   # 重新加载仍生效


def test_hot_reload_on_file_change(tmp_path):
    p = _yaml(tmp_path, "policy:\n  model:\n    retries: 1\n")
    perf = Policy(p, environ={})
    assert perf.model.retries == 1
    time.sleep(0.01)
    p.write_text("policy:\n  model:\n    retries: 5\n", encoding="utf-8")
    v = perf.version
    assert perf.model.retries == 5
    assert perf.version > v


def test_backoff_is_exponential_and_capped():
    perf = Policy(None, environ={})
    perf.set_many({"net.backoff_base": 0.1, "net.backoff_max": 1.0, "net.jitter": 0.0}, persist=False)
    waits = [perf.sleep_backoff("net", n) for n in (1, 2, 3, 4, 5)]
    assert waits == [0.1, 0.2, 0.4, 0.8, 1.0]


def test_suggest_from_observed_latency():
    perf = Policy(None, environ={})
    for _ in range(10):
        perf.record("model", 1.0, ok=True)
    tips = {t["key"] for t in perf.suggest()}
    assert "model.first_byte" in tips and "model.timeout" in tips
    st = perf.stats()["model"]
    assert st["n"] == 10 and st["p50"] == 1.0


# ---------------- 真吃策略：Provider 重试 ----------------

class _FlakyHandler(BaseHTTPRequestHandler):
    fails_before_ok = 2
    hits = 0
    delay = 0.0

    def do_POST(self):  # noqa: N802
        cls = type(self)
        cls.hits += 1
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        if cls.delay:
            time.sleep(cls.delay)
        if cls.hits <= cls.fails_before_ok:
            self.send_response(503)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"boom"}')
            return
        body = {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):  # 安静
        return


@pytest.fixture()
def flaky_server():
    handler = type("H", (_FlakyHandler,), {"hits": 0, "fails_before_ok": 2, "delay": 0.0})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", handler
    srv.shutdown()


def test_provider_without_policy_gives_up_old_behaviour(flaky_server):
    base, handler = flaky_server
    handler.hits = 0
    handler.fails_before_ok = 2

    async def go():
        prov = OpenAICompatProvider(base_url=base, model="m")
        with pytest.raises(httpx.HTTPStatusError):
            await prov.complete([{"role": "user", "content": "hi"}], [])
        await prov.aclose()
        return handler.hits

    hits = asyncio.run(go())
    assert hits == 1  # 旧写死行为：不重试，一次 503 直接失败


def test_provider_turbo_retries_through_503(flaky_server):
    base, handler = flaky_server
    handler.hits = 0
    handler.fails_before_ok = 2
    perf = Policy(None, environ={})
    perf.preset("turbo", persist=False)

    async def go():
        prov = OpenAICompatProvider(base_url=base, model="m", policy=perf)
        started = time.perf_counter()
        reply = await prov.complete([{"role": "user", "content": "hi"}], [])
        await prov.aclose()
        return reply, time.perf_counter() - started, handler.hits

    reply, elapsed, hits = asyncio.run(go())
    assert reply["content"] == "ok" and hits == 3
    assert elapsed < 1.0  # 两次退避 0.08/0.16 量级，不是线性放大


# ---------------- 真吃策略：Agent 模型级重试 ----------------

def test_agent_retry_uses_policy(monkeypatch):
    from doubao_agent.core.agent import Agent

    class FlakyProvider:
        def __init__(self):
            self.n = 0

        async def complete(self, messages, tools):
            self.n += 1
            if self.n <= 2:
                raise httpx.ConnectError("boom")
            return {"content": "ok", "tool_calls": [], "usage": {}}

        async def aclose(self):
            return None

    perf = Policy(None, environ={})
    perf.set_many({"model.retries": 2, "model.backoff_base": 0.01, "model.jitter": 0}, persist=False)
    prov = FlakyProvider()
    agent = Agent(
        registry=None, provider=prov, policy=None, sessions=None, audit=None, memory=None,
        perf=perf,
    )
    out = asyncio.run(agent._complete([{"role": "user", "content": "hi"}], []))
    assert out["content"] == "ok" and prov.n == 3


# ---------------- 真吃策略：A2A 轮询 ----------------

def test_a2a_poll_interval_from_policy():
    perf = Policy(None, environ={})
    perf.set_many({"model.poll_interval": 0.05, "model.poll_max": 0.2}, persist=False)
    client = A2AClient("http://127.0.0.1:1", allow=["127.0.0.1"], policy=perf)
    polls = {"n": 0}
    started = time.monotonic()

    async def fake_get(task_id):
        polls["n"] += 1
        state = "completed" if time.monotonic() - started > 0.3 else "working"
        return {"status": {"state": state}}

    client.get = fake_get  # type: ignore[assignment]

    async def go():
        await client.wait("t1", max_wait=2.0)
        await client.aclose()

    asyncio.run(go())
    # 0.05s 级轮询：0.3s 内至少命中 3 次；写死 2s 的老逻辑只会命中 1 次
    assert polls["n"] >= 3
    assert time.monotonic() - started < 0.8
