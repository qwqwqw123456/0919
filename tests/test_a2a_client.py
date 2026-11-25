from __future__ import annotations

import json

import httpx
import pytest

from doubao_agent.core.a2a_client import (
    A2AClient,
    A2AError,
    check_endpoint,
    summarize_card,
    summarize_task,
    task_text,
)
from doubao_agent.sources.a2a import A2ASource
from doubao_agent.types import ToolCall


CARD = {
    "protocolVersion": "0.3.0",
    "name": "peer",
    "description": "远端测试 Agent",
    "url": "http://127.0.0.1:9999/a2a",
    "version": "0.3.0",
    "capabilities": {"streaming": True, "pushNotifications": False},
    "skills": [{"id": "sys_status", "name": "system.status"}, {"id": "scan", "name": "scan"}],
}


def test_endpoint_default_loopback_only() -> None:
    assert check_endpoint("http://127.0.0.1:8799/a2a").endswith("/a2a")
    assert check_endpoint("http://127.0.0.1:8799").endswith("/a2a")
    with pytest.raises(A2AError, match="白名单"):
        check_endpoint("https://evil.example.com/a2a")
    assert check_endpoint("https://evil.example.com/a2a", ["evil.example.com"]).startswith("https://")


def test_endpoint_rejects_bad_url() -> None:
    with pytest.raises(A2AError):
        check_endpoint("ftp://127.0.0.1/a2a")
    with pytest.raises(A2AError, match="用户名"):
        check_endpoint("http://user:pw@127.0.0.1/a2a")
    # host:port 精确匹配：端口不符要拒
    with pytest.raises(A2AError):
        check_endpoint("http://127.0.0.1:9999/a2a", ["127.0.0.1:8799"])
    assert check_endpoint("http://127.0.0.1:8799/a2a", ["127.0.0.1:8799"])


def _handler(state: dict, polls_before_done: int = 2):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(200, json=CARD)
        body = json.loads(request.content or b"{}")
        method = body.get("method")
        if method == "message/send":
            blocking = bool((body.get("params") or {}).get("configuration", {}).get("blocking"))
            task = {
                "kind": "task",
                "id": "t1",
                "contextId": "c1",
                "status": {"state": "completed" if blocking else "working"},
                "artifacts": ([{"artifactId": "a1", "parts": [{"kind": "text", "text": "4"}]}] if blocking else []),
            }
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": task})
        if method == "tasks/get":
            if str((body.get("params") or {}).get("id") or "") == "missing":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {"code": -32001, "message": "task 不存在"},
                    },
                )
            state["polls"] = state.get("polls", 0) + 1
            done = state["polls"] >= polls_before_done
            task = {
                "kind": "task",
                "id": "t1",
                "contextId": "c1",
                "status": {"state": "completed" if done else "working"},
                "artifacts": [{"artifactId": "a1", "parts": [{"kind": "text", "text": "数完了：4"}]}]
                if done
                else [],
            }
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": task})
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32001, "message": "task 不存在"}},
        )

    return handle


@pytest.mark.asyncio
async def test_card_and_summary() -> None:
    client = A2AClient("http://127.0.0.1:9999/a2a", transport=httpx.MockTransport(_handler({})))
    try:
        card = await client.card()
        assert card["name"] == "peer"
        summary = summarize_card(card)
        assert "peer" in summary and "system.status" in summary and "流式：是" in summary
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_send_blocking_returns_text() -> None:
    client = A2AClient("http://127.0.0.1:9999/a2a", transport=httpx.MockTransport(_handler({})))
    try:
        task = await client.send("2+2 等于几？", blocking=True)
        assert task["status"]["state"] == "completed"
        assert task_text(task) == "4"
        assert "状态 completed" in summarize_task(task)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_send_nonblocking_polls_until_done() -> None:
    state: dict = {}
    client = A2AClient("http://127.0.0.1:9999/a2a", transport=httpx.MockTransport(_handler(state, 3)))
    try:
        task = await client.send("数一下", blocking=False, poll_interval=0.05)
        assert task["status"]["state"] == "completed"
        assert task_text(task) == "数完了：4"
        assert state["polls"] >= 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_error_is_raised_with_message() -> None:
    client = A2AClient("http://127.0.0.1:9999/a2a", transport=httpx.MockTransport(_handler({})))
    try:
        with pytest.raises(A2AError, match="task 不存在"):
            await client.get("missing")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_wait_timeout() -> None:
    client = A2AClient("http://127.0.0.1:9999/a2a", transport=httpx.MockTransport(_handler({}, 999)))
    try:
        with pytest.raises(A2AError, match="等待超时"):
            await client.wait("t1", poll_interval=0.05, max_wait=0.2)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_source_rejects_off_whitelist_without_network() -> None:
    source = A2ASource(allow=["127.0.0.1"])
    result = await source.call_tool(
        ToolCall(name="send", arguments={"url": "http://evil.example.com/a2a", "task": "hi"}, call_id="c1")
    )
    assert result.ok is False
    assert "白名单" in (result.error or "")
