from __future__ import annotations

import os
import platform
import resource
import shutil
from pathlib import Path
from typing import Any

from ..types import Risk, SourceKind, ToolCall, ToolResult, ToolSpec


def _meminfo() -> dict[str, int]:
    """解析 /proc/meminfo，返回 kB 为单位的键值；非 Linux 或读不到时为空。"""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    info[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        return info
    return info


def _collect(path: str) -> dict[str, Any]:
    mem = _meminfo()
    total_kb = mem.get("MemTotal")
    avail_kb = mem.get("MemAvailable")
    du = shutil.disk_usage(path) if path and os.path.isdir(path) else None
    # ru_maxrss 在 Linux 单位为 KB，macOS 为字节；这里按 KB 标注，跨平台仅作参考。
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_kb": {"total": total_kb, "available": avail_kb},
        "process_rss_kb": rss_kb,
    }
    if du is not None:
        metrics["disk"] = {"path": path, "total": du.total, "used": du.used, "free": du.free}
    return metrics


def _format(metrics: dict[str, Any]) -> str:
    mem = metrics["memory_kb"]
    lines = [f"平台：{metrics['platform']}", f"CPU 核数：{metrics['cpu_count']}"]
    if mem["total"] is not None:
        avail = mem["available"] if mem["available"] is not None else "未知"
        lines.append(f"内存：总 {mem['total']} KB / 可用 {avail} KB")
    else:
        lines.append("内存：当前平台不可读 /proc/meminfo")
    lines.append(f"本进程 RSS：{metrics['process_rss_kb']} KB")
    disk = metrics.get("disk")
    if disk:
        lines.append(
            f"磁盘（{disk['path']}）：总 {disk['total']} 字节 / 已用 {disk['used']} / 空闲 {disk['free']}"
        )
    return "\n".join(lines)


class SystemSource:
    """只读系统指标通道：供规划判断资源与可行性，不依赖 shell。"""

    name = "system"
    kind = SourceKind.BUILTIN

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = str(workspace)

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="status",
                description=(
                    "读取当前运行环境的只读系统指标：CPU 核数、内存总量/可用、本进程内存、磁盘用量。"
                    "判断任务可行性、资源是否紧张时先查，不要凭感觉估算。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "要查磁盘用量的目录，默认工作区；可选",
                        }
                    },
                },
                kind=self.kind,
                risk=Risk.READ,
                idempotent=True,
                source=self.name,
            )
        ]

    async def call_tool(self, call: ToolCall) -> ToolResult:
        if call.name != "status":
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown system tool: {call.name}")
        args = call.arguments or {}
        path = str(args.get("path") or self.workspace)
        try:
            metrics = _collect(path)
        except Exception as exc:  # noqa: BLE001 - 采集失败降级为错误结果，不炸主链路
            return ToolResult(
                call_id=call.call_id, ok=False, error=f"{type(exc).__name__}: {exc}"
            )
        return ToolResult(call_id=call.call_id, ok=True, content=_format(metrics), meta={"metrics": metrics})

    async def health(self) -> tuple[bool, str]:
        metrics = _collect(self.workspace)
        return True, f"cpu={metrics['cpu_count']} · rss={metrics['process_rss_kb']} KB"

    async def aclose(self) -> None:
        return None
