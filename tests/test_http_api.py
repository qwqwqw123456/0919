from __future__ import annotations

from doubao_agent.sources.http_api import (
    _endpoint_schema,
    _parse_declarative,
    _parse_openapi,
)


def test_parse_openapi_extracts_endpoints_and_dedupes_params() -> None:
    doc = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.example.com/v1"}],
        "paths": {
            "/users/{id}": {
                "get": {
                    "operationId": "getUser",
                    "summary": "取用户",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "q", "in": "query", "schema": {"type": "string"}},
                    ],
                },
                "post": {
                    "requestBody": {},
                    "parameters": [{"name": "q", "in": "query"}],
                },
            }
        },
    }
    base_url, endpoints = _parse_openapi(doc)
    assert base_url == "https://api.example.com/v1"
    by_name = {ep["name"]: ep for ep in endpoints}
    assert by_name["getUser"]["method"] == "GET"
    assert by_name["getUser"]["has_body"] is False
    post = by_name["post_users_id"]
    assert post["method"] == "POST"
    assert post["has_body"] is True
    # 路径参数保留在 params 里，供 call_tool 替换占位
    assert any(p["name"] == "id" and p["in"] == "path" for p in by_name["getUser"]["params"])


def test_parse_declarative_reads_endpoints() -> None:
    doc = {
        "base_url": "https://x.example.com",
        "endpoints": [
            {"name": "ping", "method": "get", "path": "/ping", "params": [{"name": "host", "in": "query"}]}
        ],
    }
    base_url, endpoints = _parse_declarative("crm", doc)
    assert base_url == "https://x.example.com"
    assert endpoints[0]["name"] == "ping"
    assert endpoints[0]["method"] == "GET"


def test_endpoint_schema_collects_required_and_body() -> None:
    schema = _endpoint_schema(
        [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
        has_body=True,
    )
    assert schema["required"] == ["id"]
    assert "body" in schema["properties"]
    assert schema["properties"]["id"]["type"] == "integer"
