from __future__ import annotations

import httpx
import pytest

from doubao_agent.config import Config, load_config
from doubao_agent.core.knowledge import KnowledgeStore
from doubao_agent.providers.embedding import BridgeEmbedding
from doubao_agent.sources.bridge import HttpHostBridge


def _bridge(handler) -> HttpHostBridge:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpHostBridge("http://host.test", "t", client=client)


async def test_bridge_embedding_reads_capabilities_and_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"capabilities": [{"name": "embed"}]})
        return httpx.Response(
            200, json={"result": {"data": [{"embedding": [0.1, 0.2, 0.3]}]}}
        )

    embedding = BridgeEmbedding(_bridge(handler))
    assert await embedding.available() is True
    assert await embedding.embed("你好") == [0.1, 0.2, 0.3]


async def test_bridge_embedding_unavailable_without_capability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": [{"name": "browser.open"}]})

    embedding = BridgeEmbedding(_bridge(handler))
    assert await embedding.available() is False


async def test_bridge_embedding_rejects_unparsable_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(200, json={"capabilities": [{"name": "embed"}]})
        return httpx.Response(200, json={"result": {"unexpected": "shape"}})

    with pytest.raises(ValueError):
        await BridgeEmbedding(_bridge(handler)).embed("x")


async def test_knowledge_asearch_uses_async_embed_and_caches(tmp_path) -> None:
    calls: list[str] = []

    async def aembed(text: str) -> list[float]:
        calls.append(text)
        return [1.0, float(len(text))]

    store = KnowledgeStore(tmp_path / "k.db", aembed=aembed)
    store.add_text("制度", "第一段说明。\n\n第二段说明。")

    await store.asearch("第一段")
    first = len(calls)
    await store.asearch("第二段")
    # 第二次检索只多算 query 向量，分块向量命中缓存
    assert len(calls) == first + 1


async def test_knowledge_asearch_degrades_when_embedding_fails(tmp_path) -> None:
    async def aembed(text: str) -> list[float]:
        raise RuntimeError("bridge down")

    store = KnowledgeStore(tmp_path / "k.db", aembed=aembed)
    store.add_text("制度", "重要结论：A 等于 B")

    hits = await store.asearch("A 等于 B")
    assert hits and "A" in hits[0]["text"]


def test_config_workspace_defaults_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("AGENT_DB_PATH", raising=False)
    monkeypatch.delenv("AGENT_RUNS_DIR", raising=False)
    monkeypatch.delenv("AGENT_SKILLS_DIR", raising=False)

    cfg = Config()
    assert cfg.workspace == tmp_path.resolve()
    assert cfg.db_path == tmp_path.resolve() / ".agent" / "agent.db"
    assert cfg.runs_dir == tmp_path.resolve() / "runs"
    assert cfg.skills_dir == tmp_path.resolve() / "skills"


def test_config_embedding_capability_env_override(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text("embedding:\n  enabled: true\n  capability: embed\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_EMBEDDING_CAPABILITY", "model.embed")

    cfg = load_config(cfg_path)
    assert cfg.embedding["capability"] == "model.embed"
