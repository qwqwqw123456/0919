from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec
from ..utils import dumps, infer_risk
from .base import SourceError

_METHODS = ("get", "post", "put", "patch", "delete")


def _endpoint_schema(params: list[dict[str, Any]], has_body: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        schema = dict(p.get("schema") or {"type": "string"})
        if p.get("description"):
            schema.setdefault("description", p["description"])
        properties[p["name"]] = schema
        if p.get("required"):
            required.append(p["name"])
    if has_body:
        properties["body"] = {
            "type": "object",
            "description": "JSON request body",
        }
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parse_declarative(name: str, doc: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    base_url = doc.get("base_url", "")
    endpoints = []
    for ep in doc.get("endpoints", []):
        endpoints.append(
            {
                "name": ep["name"],
                "method": ep.get("method", "GET").upper(),
                "path": ep["path"],
                "description": ep.get("description", ""),
                "params": ep.get("params", []),
                "has_body": bool(ep.get("has_body") or ep.get("body")),
                "risk": ep.get("risk"),
            }
        )
    return base_url, endpoints


def _parse_openapi(doc: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    servers = doc.get("servers") or []
    base_url = servers[0].get("url", "") if servers else ""
    endpoints = []
    for path, item in (doc.get("paths") or {}).items():
        for method in _METHODS:
            op = item.get(method)
            if not op:
                continue
            raw_name = op.get("operationId") or f"{method}_{path.strip('/').replace('/', '_')}"
            name = raw_name.replace("{", "").replace("}", "").replace("-", "_")
            params = []
            seen = set()
            for p in (item.get("parameters") or []) + (op.get("parameters") or []):
                key = (p.get("name"), p.get("in"))
                if key in seen:
                    continue
                seen.add(key)
                params.append(
                    {
                        "name": p.get("name"),
                        "in": p.get("in", "query"),
                        "required": p.get("required", False),
                        "schema": p.get("schema") or {"type": "string"},
                        "description": p.get("description", ""),
                    }
                )
            endpoints.append(
                {
                    "name": name,
                    "method": method.upper(),
                    "path": path,
                    "description": op.get("summary") or op.get("description", ""),
                    "params": params,
                    "has_body": "requestBody" in op,
                    "risk": None,
                }
            )
    return base_url, endpoints


class ApiSource:
    """Turns any HTTP interface (declarative or OpenAPI) into unified tools."""

    kind = SourceKind.API

    def __init__(self, name: str, spec: dict[str, Any], base_url: str,
                 endpoints: list[dict[str, Any]]) -> None:
        self.name = name
        self.spec = spec
        self.base_url = base_url.rstrip("/")
        self.endpoints = {ep["name"]: ep for ep in endpoints}
        self._client: httpx.AsyncClient | None = None
        self._specs = [self._to_spec(ep) for ep in endpoints]

    @classmethod
    def from_file(cls, name: str, path: str | Path, base_url: str | None = None) -> "ApiSource":
        import json

        import yaml

        raw = Path(path).read_text("utf-8")
        doc = yaml.safe_load(raw) if path.endswith((".yaml", ".yml")) else json.loads(raw)
        if not isinstance(doc, dict):
            raise SourceError(f"invalid API spec: {path}")

        if "openapi" in doc or "swagger" in doc:
            url, endpoints = _parse_openapi(doc)
        elif "spec" in doc:  # wrapper: {name, spec: <openapi path or inline>, base_url}
            inner = doc["spec"]
            inner_doc = yaml.safe_load(Path(inner).read_text("utf-8")) if isinstance(inner, str) else inner
            url, endpoints = _parse_openapi(inner_doc)
            base_url = base_url or doc.get("base_url")
        else:
            url, endpoints = _parse_declarative(name, doc)

        return cls(name=name, spec=doc, base_url=base_url or url, endpoints=endpoints)

    def _to_spec(self, ep: dict[str, Any]) -> ToolSpec:
        desc = ep.get("description") or f"{ep['method']} {ep['path']}"
        raw_risk = ep.get("risk")
        try:
            risk = Risk(str(raw_risk).upper()) if raw_risk else infer_risk(ep["name"], desc)
        except ValueError:
            risk = infer_risk(ep["name"], desc)
        return ToolSpec(
            name=ep["name"],
            description=desc,
            input_schema=_endpoint_schema(ep.get("params", []), ep.get("has_body", False)),
            kind=SourceKind.API,
            risk=risk,
            idempotent=ep["method"] in ("GET", "HEAD", "PUT", "DELETE"),
            source=self.name,
        )

    def _auth_headers(self) -> dict[str, str]:
        auth = self.spec.get("auth") or {}
        if auth.get("type") == "bearer":
            token = os.environ.get(auth.get("token_env", ""), "") or auth.get("token", "")
            return {"Authorization": f"Bearer {token}"} if token else {}
        if auth.get("type") == "header" and auth.get("name"):
            value = os.environ.get(auth.get("value_env", ""), "") or auth.get("value", "")
            return {auth["name"]: value} if value else {}
        return {}

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._specs)

    async def call_tool(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        ep = self.endpoints.get(call.name)
        if ep is None:
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown endpoint {call.name}")

        path = ep["path"]
        query: dict[str, Any] = {}
        headers: dict[str, str] = {**(self.spec.get("headers") or {}), **self._auth_headers()}
        body = None

        for p in ep.get("params", []):
            key = p["name"]
            if key not in call.arguments:
                continue
            value = call.arguments[key]
            where = p.get("in", "query")
            if where == "path":
                path = path.replace("{" + key + "}", str(value))
            elif where == "header":
                headers[key] = str(value)
            else:
                query[key] = value

        if ep.get("has_body"):
            body = call.arguments.get("body", {k: v for k, v in call.arguments.items() if k != "body"})

        url = f"{self.base_url}{path}"
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=60.0, trust_env=True)
            response = await self._client.request(
                ep["method"], url, params=query, headers=headers, json=body
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            payload: Any = response.json()
        except Exception:  # noqa: BLE001 - non-JSON body is still a valid observation
            payload = response.text

        ok = response.is_success
        return ToolResult(
            call_id=call.call_id,
            ok=ok,
            content=dumps(payload),
            error=None if ok else f"HTTP {response.status_code}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            meta={"status": response.status_code},
        )

    async def health(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "no base_url"
        return True, f"{len(self._specs)} endpoints @ {self.base_url}"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
