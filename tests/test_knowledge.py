from __future__ import annotations

from doubao_agent.types import ToolCall


def test_knowledge_chunking_and_search(tmp_path) -> None:
    from doubao_agent.core.knowledge import KnowledgeStore

    store = KnowledgeStore(tmp_path / "k.db")
    doc_id = store.add_text(
        "报销制度",
        "公司报销周期为每月 25 日。\n\n差旅费需提前申请，住宿标准每晚不超过 500 元。\n\n餐饮补贴每人每天 80 元。",
    )

    assert doc_id > 0
    hits = store.search("报销周期是什么时候", limit=3)
    assert hits and hits[0]["doc_id"] == doc_id
    assert "25" in hits[0]["text"]

    # 语义近似：没有完全相同的词，bigram 重叠也能命中相关块
    hits2 = store.search("出差住宿标准")
    assert any("住宿" in hit["text"] for hit in hits2)

    assert store.stats()["documents"] == 1
    assert store.remove(doc_id) is True
    assert store.list_documents() == []


async def test_knowledge_source_tools(tmp_path) -> None:
    from doubao_agent.core.knowledge import KnowledgeStore
    from doubao_agent.sources.knowledge import KnowledgeSource

    store = KnowledgeStore(tmp_path / "k.db")
    source = KnowledgeSource(store)

    specs = await source.list_tools()
    assert {"search", "add"} <= {spec.name for spec in specs}

    add = await source.call_tool(
        ToolCall(name="add", arguments={"title": "T", "text": "重要结论：A 等于 B"}, call_id="a")
    )
    assert add.ok is True

    search = await source.call_tool(
        ToolCall(name="search", arguments={"query": "A 和 B 的关系"}, call_id="s")
    )
    assert search.ok is True and "A" in search.content
