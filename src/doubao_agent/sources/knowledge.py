from __future__ import annotations

from typing import Any

from ..core.knowledge import KnowledgeStore
from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec


class KnowledgeSource:
    """知识库通道：agent 可主动检索知识，也可把结论沉淀回知识库。"""

    name = "knowledge"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search",
                description=(
                    "在知识库中检索相关资料（语义化 + 关键词），返回最相关的文档片段。"
                    "遇到需要背景资料、制度规范、历史文档时先用它查，不要凭空编造。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题或关键词"},
                        "limit": {"type": "integer", "description": "返回片段数，默认 4"},
                    },
                    "required": ["query"],
                },
                kind=SourceKind.KNOWLEDGE,
                risk=Risk.READ,
                source=self.name,
            ),
            ToolSpec(
                name="add",
                description=(
                    "把一段内容写入知识库，供以后检索复用。适合沉淀结论、整理好的资料、"
                    "需要长期保留的文档。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "标题"},
                        "text": {"type": "string", "description": "正文内容"},
                        "source": {"type": "string", "description": "来源说明，可选"},
                    },
                    "required": ["title", "text"],
                },
                kind=SourceKind.KNOWLEDGE,
                risk=Risk.REVERSIBLE,
                idempotent=False,
                source=self.name,
            ),
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        args = call.arguments or {}
        if call.name == "search":
            query = str(args.get("query", "")).strip()
            if not query:
                return ToolResult(call_id=call.call_id, ok=False, error="query is required")
            limit = int(args.get("limit") or 4)
            hits = self.store.search(query, limit=limit)
            if not hits:
                return ToolResult(call_id=call.call_id, ok=True, content="（知识库中没有找到相关内容）")
            lines = []
            for hit in hits:
                lines.append(
                    f"## {hit['title']}（#{hit['doc_id']} · 相关度 {hit['score']}）\n{hit['text']}"
                )
            return ToolResult(call_id=call.call_id, ok=True, content="\n\n".join(lines))

        if call.name == "add":
            title = str(args.get("title", "")).strip()
            text = str(args.get("text", "")).strip()
            if not title or not text:
                return ToolResult(call_id=call.call_id, ok=False, error="title and text are required")
            try:
                doc_id = self.store.add_text(title, text, source=str(args.get("source", "")))
            except ValueError as exc:
                return ToolResult(call_id=call.call_id, ok=False, error=str(exc))
            return ToolResult(call_id=call.call_id, ok=True, content=f"已写入知识库 #{doc_id}")

        return ToolResult(call_id=call.call_id, ok=False, error=f"unknown knowledge tool: {call.name}")

    async def health(self) -> tuple[bool, str]:
        stats = self.store.stats()
        mode = "vector" if self.store.embed is not None else "hybrid"
        return True, f"{stats['documents']} docs / {stats['chunks']} chunks · {mode}"

    async def aclose(self) -> None:
        # Store 生命周期由 Runtime 统一管理，这里不关闭，避免双重 close。
        return None
