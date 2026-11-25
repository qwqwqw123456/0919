from __future__ import annotations

import json

from fastapi.testclient import TestClient

from doubao_agent.config import Config
from doubao_agent.server.app import create_app


def _config(tmp_path, token: str = "") -> Config:
    cfg = Config()
    cfg.workspace = tmp_path
    cfg.runs_dir = tmp_path / "runs"
    cfg.db_path = tmp_path / ".agent" / "agent.db"
    cfg.skills_dir = tmp_path / "skills"
    cfg.scheduler_enabled = False
    cfg.mcp_auto_discover = False
    cfg.model = {"provider": "mock"}
    cfg.auth_token = token
    cfg.ensure_dirs()
    return cfg


def _msg(text: str, **extra) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "m1",
                "parts": [{"kind": "text", "text": text}],
                **extra,
            },
            "configuration": {"blocking": True},
        },
    }


def test_agent_card_minimal(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        card = client.get("/.well-known/agent-card.json").json()
        assert card["protocolVersion"] == "0.3.0"
        assert card["url"].endswith("/a2a")
        assert card["capabilities"]["streaming"] is True
        assert card["skills"], "卡片应带上工具能力清单"
        text = json.dumps(card)
        assert "token" not in text.lower()
        # 兼容旧路径
        assert client.get("/agent.json").json()["name"] == card["name"]


def test_agent_card_hides_secret_but_declares_scheme(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        card = client.get("/.well-known/agent-card.json").json()
        assert card["securitySchemes"]["bearer"]["scheme"] == "bearer"
        assert "s3cret" not in json.dumps(card)


def test_message_send_blocking_completes(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        body = client.post("/a2a", json=_msg("你好，介绍一下你自己")).json()
        task = body["result"]
        assert task["kind"] == "task"
        assert task["status"]["state"] == "completed"
        assert task["artifacts"], "完成的任务必须带回 artifacts"
        assert task["artifacts"][0]["parts"][0]["kind"] == "text"
        assert task["history"][0]["role"] == "user"
        # task id 就是会话 id，可直接用 tasks/get 续查
        got = client.post(
            "/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task["id"]}}
        ).json()
        assert got["result"]["id"] == task["id"]
        assert got["result"]["status"]["state"] == "completed"
        # 已结束的任务不能取消
        cancelled = client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 3, "method": "tasks/cancel", "params": {"id": task["id"]}},
        ).json()
        assert cancelled["error"]["code"] == -32002


def test_tasks_get_unknown(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        body = client.post(
            "/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "nope"}}
        ).json()
        assert body["error"]["code"] == -32001


def test_message_requires_text_parts(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        bad = _msg("x")
        bad["params"]["message"]["parts"] = [{"kind": "text", "text": "  "}]
        body = client.post("/a2a", json=bad).json()
        assert body["error"]["code"] == -32602


def test_unknown_method_rejected(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        body = client.post(
            "/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "foo/bar", "params": {}}
        ).json()
        assert body["error"]["code"] == -32601
        broken = client.post("/a2a", content=b"{not json")
        assert broken.json()["error"]["code"] == -32700


def test_stream_emits_status_then_task(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        payload = _msg("数一下")
        payload["method"] = "message/stream"
        with client.stream("POST", "/a2a", json=payload) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            chunks = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
        assert chunks, "流式响应不能为空"
        assert chunks[-1]["result"]["kind"] == "task"
        assert chunks[-1]["result"]["status"]["state"] == "completed"
        mid = [c for c in chunks[:-1] if c["result"]["kind"] == "status-update"]
        assert mid and mid[0]["result"]["taskId"] == chunks[-1]["result"]["id"]
        assert mid[0]["result"]["final"] is False


def test_endpoint_requires_token(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        assert client.post("/a2a", json=_msg("hi")).status_code == 401
        ok = client.post("/a2a", json=_msg("hi"), headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert ok.json()["result"]["status"]["state"] == "completed"


def test_data_and_file_parts_are_texted() -> None:
    from doubao_agent.core.a2a import parts_to_text

    text = parts_to_text(
        [
            {"kind": "text", "text": "看下这份"},
            {"kind": "data", "data": {"a": 1}},
            {"kind": "file", "file": {"name": "报告.pdf", "uri": "file:///tmp/a.pdf"}},
        ]
    )
    assert "看下这份" in text
    assert '{"a": 1}' in text
    assert "报告.pdf" in text


def test_async_send_then_poll(tmp_path) -> None:
    import time

    with TestClient(create_app(_config(tmp_path))) as client:
        payload = _msg("你好")
        payload["params"]["configuration"]["blocking"] = False
        task = client.post("/a2a", json=payload).json()["result"]
        assert task["status"]["state"] == "working"
        state = "working"
        for _ in range(50):
            got = client.post(
                "/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task["id"]}}
            ).json()["result"]
            state = got["status"]["state"]
            if state != "working":
                break
            time.sleep(0.05)
        assert state == "completed"
        assert got["artifacts"]
