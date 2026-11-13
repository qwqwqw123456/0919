from __future__ import annotations

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


def test_metrics_and_trace_endpoints(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        metrics = client.get("/api/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["spans"] == 0
        trace = client.get("/api/sessions/demo/trace")
        assert trace.status_code == 200
        assert trace.json() == []


def test_run_endpoint_rejects_without_token(tmp_path) -> None:
    with TestClient(create_app(_config(tmp_path, "s3cret"))) as client:
        response = client.post("/api/run", json={"task": "查一下资料"})
        assert response.status_code == 401
