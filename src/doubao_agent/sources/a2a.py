from __future__ import annotations

from typing import Any

from ..core.a2a_client import A2AClient, A2AError, summarize_card, summarize_task
from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec


class A2ASource:
    """把 A2A 客户端暴露成工具：认对端能力、发任务、等结果、取消。"""

    name = "a2a"
    kind = SourceKind.BUILTIN

    def __init__(self, allow: list[str] | None = None, token: str = "") -> None:
        self.allow = list(allow or [])
        self.token = token

    def _client(self, url: str, timeout: float) -> A2AClient:
        return A2AClient(url, token=self.token, allow=self.allow or None, timeout=timeout)

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="discover",
                description=(
                    "读取远端 A2A Agent 的 Agent Card，确认它叫什么、支持什么（是否流式、有哪些技能）。"
                    "在把任务交给对方之前先查，避免把活派给不认识的对端。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "对端 A2A 地址，如 http://127.0.0.1:8799/a2a"}},
                    "required": ["url"],
                },
                kind=self.kind,
                risk=Risk.READ,
                idempotent=True,
                source=self.name,
            ),
            ToolSpec(
                name="send",
                description=(
                    "把一个任务发给远端 A2A Agent 执行，并等待其终态后取回文本结果。"
                    "注意：任务文本会发给对端，对端可能据此采取动作；只发给白名单内、你信任的地址。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "对端 A2A 地址"},
                        "task": {"type": "string", "description": "要交给对端执行的任务描述"},
                        "timeout": {
                            "type": "number",
                            "description": "最长等待秒数，默认 600",
                            "default": 600,
                        },
                    },
                    "required": ["url", "task"],
                },
                kind=self.kind,
                risk=Risk.REVERSIBLE,
                idempotent=False,
                source=self.name,
            ),
            ToolSpec(
                name="cancel",
                description="取消远端 A2A Agent 上仍在运行的某个 task（需要 task id）。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "对端 A2A 地址"},
                        "task_id": {"type": "string", "description": "要取消的 task id"},
                    },
                    "required": ["url", "task_id"],
                },
                kind=self.kind,
                risk=Risk.REVERSIBLE,
                idempotent=True,
                source=self.name,
            ),
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        args: dict[str, Any] = call.arguments or {}
        url = str(args.get("url") or "").strip()
        if not url:
            return ToolResult(call_id=call.call_id, ok=False, error="url 不能为空")
        timeout = float(args.get("timeout") or (120.0 if call.name == "discover" else 600.0))
        client: A2AClient | None = None
        try:
            client = self._client(url, timeout)
            if call.name == "discover":
                card = await client.card()
                return ToolResult(
                    call_id=call.call_id,
                    ok=True,
                    content=summarize_card(card),
                    meta={"card": card},
                )
            if call.name == "send":
                task = str(args.get("task") or "").strip()
                if not task:
                    return ToolResult(call_id=call.call_id, ok=False, error="task 不能为空")
                result = await client.send(task, wait=True, max_wait=timeout)
                state = str(result.get("status", {}).get("state") or "unknown")
                return ToolResult(
                    call_id=call.call_id,
                    ok=state != "failed",
                    content=summarize_task(result),
                    error="" if state != "failed" else summarize_task(result),
                    meta={"task_id": result.get("id"), "state": state, "url": client.endpoint},
                )
            if call.name == "cancel":
                task_id = str(args.get("task_id") or "").strip()
                if not task_id:
                    return ToolResult(call_id=call.call_id, ok=False, error="task_id 不能为空")
                result = await client.cancel(task_id)
                return ToolResult(
                    call_id=call.call_id,
                    ok=True,
                    content=summarize_task(result),
                    meta={"task_id": result.get("id")},
                )
        except A2AError as exc:
            return ToolResult(call_id=call.call_id, ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - 工具不能把异常抛进主链路
            return ToolResult(call_id=call.call_id, ok=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            if client is not None:
                await client.aclose()
        return ToolResult(call_id=call.call_id, ok=False, error=f"unknown a2a tool: {call.name}")

    async def health(self) -> tuple[bool, str]:
        return True, f"allow={', '.join(self.allow) or 'loopback only'}"

    async def aclose(self) -> None:
        return None
