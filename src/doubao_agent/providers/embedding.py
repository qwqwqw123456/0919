from __future__ import annotations

from typing import Any

from ..sources.bridge import HostBridge


def _vectors_of(result: Any) -> list[float] | None:
    """从宿主返回里取出单条向量，兼容几种常见形态。

    支持：[0.1, 0.2] / {"embedding": [...]} / {"vectors": [[...]]} /
    {"data": [{"embedding": [...]}]} / {"result": <上述任一>}
    """
    if isinstance(result, dict) and "result" in result:
        return _vectors_of(result["result"])
    if isinstance(result, list) and result and all(isinstance(x, (int, float)) for x in result):
        return [float(x) for x in result]
    if isinstance(result, dict):
        for key in ("embedding", "vector"):
            value = result.get(key)
            if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                return [float(x) for x in value]
        vectors = result.get("vectors")
        if isinstance(vectors, list) and vectors:
            return _vectors_of(vectors[0])
        data = result.get("data")
        if isinstance(data, list) and data:
            return _vectors_of(data[0])
    return None


class BridgeEmbedding:
    """宿主 embedding：按能力名调用桥，返回向量。

    宿主清单里没有该能力时不启用（KnowledgeStore 保持关键词检索），
    绝不本地伪造向量。
    """

    def __init__(self, bridge: HostBridge, capability: str = "embed") -> None:
        self.bridge = bridge
        self.capability = capability
        self._available: bool | None = None

    async def available(self) -> bool:
        if self._available is None:
            try:
                caps = await self.bridge.list_capabilities()
            except Exception:  # noqa: BLE001 - 发现失败按不可用处理
                caps = []
            names = {str(item.get("name")) for item in caps if isinstance(item, dict)}
            self._available = self.capability in names
        return bool(self._available)

    async def embed(self, text: str) -> list[float]:
        result = await self.bridge.invoke(self.capability, {"input": text})
        vector = _vectors_of(result)
        if vector is None:
            raise ValueError(f"embedding 响应无法解析: {type(result).__name__}")
        return vector
