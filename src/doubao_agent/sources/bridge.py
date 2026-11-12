from __future__ import annotations

from typing import Any, Protocol

import httpx


class HostBridge(Protocol):
    """Adapter to Doubao's native capabilities.

    The host may expose hundreds of capabilities; the runtime discovers them
    instead of hardcoding a list.
    """

    async def invoke(self, capability: str, payload: dict[str, Any]) -> Any: ...

    async def list_capabilities(self) -> list[dict[str, Any]]: ...

    async def aclose(self) -> None: ...


class HttpHostBridge:
    def __init__(
        self,
        url: str,
        token: str | None = None,
        capabilities_path: str = "/capabilities",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.capabilities_path = capabilities_path
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=True)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def invoke(self, capability: str, payload: dict[str, Any]) -> Any:
        response = await self._client.post(
            f"{self.url}/invoke",
            json={"capability": capability, "payload": payload},
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", data) if isinstance(data, dict) else data

    async def list_capabilities(self) -> list[dict[str, Any]]:
        """Fetch the host's capability manifest. An empty list means fall back."""
        try:
            response = await self._client.get(
                f"{self.url}{self.capabilities_path}", headers=self._headers(), timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        except Exception:  # noqa: BLE001 - discovery is best-effort
            return []
        items = data.get("capabilities") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict) and item.get("name")]

    async def aclose(self) -> None:
        await self._client.aclose()
