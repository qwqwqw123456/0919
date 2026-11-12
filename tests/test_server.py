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


def test_api_open_when_no_token_configured(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.get("/api/tools")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_api_requires_token_when_configured(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        assert client.get("/api/tools").status_code == 401
        assert client.get("/api/tools", headers={"X-Agent-Token": "wrong"}).status_code == 401
        assert client.get("/api/tools", headers={"X-Agent-Token": "s3cret"}).status_code == 200
        health = client.get("/api/health", headers={"Authorization": "Bearer s3cret"})
        assert health.status_code == 200


def test_query_token_sets_cookie(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        assert client.get("/api/tools?token=s3cret").status_code == 200
        # TestClient 自动保存 cookie，后续请求不再需要显式带头
        assert client.cookies.get("agent_token") == "s3cret"
        assert client.get("/api/tools").status_code == 200


def test_run_endpoint_rejects_without_token(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        response = client.post("/api/run", json={"task": "查一下资料"})
        assert response.status_code == 401


def _sign(token: str, body: bytes) -> str:
    import hashlib
    import hmac

    return "sha256=" + hmac.new(token.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_open_when_no_token_configured(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.post("/api/webhook", json={"task": "整理今日记录", "event": "manual"})
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert response.json()["event"] == "manual"


def test_webhook_rejects_without_token_or_signature(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        assert client.post("/api/webhook", json={"task": "任务"}).status_code == 401


def test_webhook_accepts_valid_signature(tmp_path) -> None:
    body = json.dumps({"task": "整理今日记录", "event": "push"}, ensure_ascii=False).encode("utf-8")
    headers = {
        "X-Agent-Signature": _sign("s3cret", body),
        "content-type": "application/json",
    }
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        response = client.post("/api/webhook", content=body, headers=headers)
        assert response.status_code == 200
        assert response.json()["event"] == "push"


def test_webhook_rejects_bad_signature(tmp_path) -> None:
    body = json.dumps({"task": "整理今日记录"}, ensure_ascii=False).encode("utf-8")
    headers = {"X-Agent-Signature": "sha256=deadbeef", "content-type": "application/json"}
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        assert client.post("/api/webhook", content=body, headers=headers).status_code == 401


def test_webhook_requires_task(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.post("/api/webhook", json={"event": "x"}).status_code == 400
