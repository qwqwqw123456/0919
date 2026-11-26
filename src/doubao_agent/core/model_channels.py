"""模型通道管理：列通道、切主通道、探活。

换基座不改代码，只改配置里的通道；命令不写死模型名，模型名从 /models 实测拉取，
免得凭记忆填错。
"""

from __future__ import annotations

import os
from typing import Any

import httpx

CURRENT_KEY = "model"


def mask_key(key: str) -> str:
    """凭证只显前 4 后 4，中间打码。"""
    if not key:
        return "(未配置)"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}********{key[-4:]}"


def channels(cfg: Any) -> list[dict[str, Any]]:
    """主通道 + 备用 + models: 清单，统一成一张表。"""
    rows: list[dict[str, Any]] = []
    main = dict(getattr(cfg, "model", {}) or {})
    if main:
        rows.append({"name": "main", "current": True, **main})
    fb = dict(getattr(cfg, "model_fallback", {}) or {})
    if fb:
        rows.append({"name": "fallback", "current": False, **fb})
    for item in getattr(cfg, "models", []) or []:
        if isinstance(item, dict) and item.get("name"):
            rows.append({"name": item["name"], "current": False, **item})
    return rows


def resolve_key(channel: dict[str, Any]) -> str:
    env = channel.get("api_key_env") or "OPENAI_API_KEY"
    return os.environ.get(env, "")


def find_channel(cfg: Any, name: str) -> dict[str, Any] | None:
    for row in channels(cfg):
        if row["name"] == name:
            return row
    return None


def switch_channel(config_path: str, name: str) -> dict[str, Any]:
    """把指定通道写成 agent.yaml 的主 model，原主通道沉到 model_fallback。

    只动 model / model_fallback 两段，其余配置原样保留。
    """
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    candidates = {}
    for item in data.get("models") or []:
        if isinstance(item, dict) and item.get("name"):
            candidates[item["name"]] = dict(item)
    if name in candidates:
        target = candidates[name]
    elif name == "main" and data.get("model"):
        target = dict(data["model"])
    else:
        raise KeyError(f"没有名为 {name} 的通道")

    old_main = dict(data.get("model") or {})
    payload = {k: v for k, v in target.items() if k != "name"}
    data["model"] = payload
    if old_main and old_main != payload:
        data["model_fallback"] = old_main
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return {"to": name, "model": payload.get("model"), "fallback": old_main.get("model")}


async def list_models(channel: dict[str, Any]) -> list[str]:
    key = resolve_key(channel)
    if not key:
        raise RuntimeError(f"缺少密钥：环境变量 {channel.get('api_key_env') or 'OPENAI_API_KEY'} 未设置")
    base = str(channel.get("base_url") or "").rstrip("/")
    async with httpx.AsyncClient(base_url=base, timeout=30.0, trust_env=True) as client:
        resp = await client.get("/models", headers={"Authorization": f"Bearer {key}"})
        resp.raise_for_status()
        body = resp.json()
    items = body.get("data") if isinstance(body, dict) else body
    return [str(x.get("id")) for x in (items or []) if isinstance(x, dict) and x.get("id")]


async def probe(channel: dict[str, Any], model: str | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """真发一次最小请求，返回是否通、延迟、回包摘要；不写入任何凭据。"""
    import time

    key = resolve_key(channel)
    if not key:
        return {
            "ok": False,
            "error": f"缺少密钥：{channel.get('api_key_env') or 'OPENAI_API_KEY'} 未设置",
        }
    base = str(channel.get("base_url") or "").rstrip("/")
    target = model or str(channel.get("model") or "")
    payload = {
        "model": target,
        "messages": [{"role": "user", "content": "回复两个字：就绪"}],
        "max_tokens": 32,
        "temperature": 0,
    }
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(
            base_url=base, timeout=timeout, trust_env=True
        ) as client:
            resp = await client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            )
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001 - 探活要如实报错，不吞
        return {"ok": False, "model": target, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = round(time.perf_counter() - started, 2)
    text = ""
    try:
        text = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        text = str(body)[:120]
    return {"ok": True, "model": target, "latency_s": elapsed, "reply": text.strip()[:80]}
