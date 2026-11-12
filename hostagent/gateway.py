"""宿主在线运行时网关（manifest 第 C 层）。

本地进程不持有 MCP 客户端与凭证，不直连、不伪造结果。
未配置 `HOST_RUNTIME_URL` 时，external_runtime 节点直接报错，
调用方必须把它当作「需要宿主执行」而非成功。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class DelegateUnavailable(RuntimeError):
    """宿主在线运行时不可用：本地不允许伪造结果。"""


class HostRuntimeGateway:
    def __init__(self, url: str | None = None, timeout: float = 30.0) -> None:
        self.url = (url or os.environ.get("HOST_RUNTIME_URL") or "").rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.url)

    def invoke(self, action: str, params: dict[str, Any]) -> Any:
        if not self.available:
            raise DelegateUnavailable(
                "宿主在线运行时未配置（HOST_RUNTIME_URL）：本地不持凭证、不直连、不伪造结果，"
                f"action={action} 需由宿主执行"
            )
        body = json.dumps({"action": action, "params": params}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/invoke",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise DelegateUnavailable(f"宿主运行时返回 HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DelegateUnavailable(f"宿主运行时不可达: {exc}") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DelegateUnavailable("宿主运行时返回非 JSON") from exc
        if isinstance(data, dict) and data.get("ok") is False:
            raise DelegateUnavailable(str(data.get("error") or "宿主运行时返回失败"))
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data
