from __future__ import annotations

import json
from typing import Any

from .types import Risk

_SENIOR = (
    "sudo", "chmod", "chown", "mkfs", "format_disk", "shutdown", "reboot",
    "system_config", "registry_edit", "rm_rf", "drop_database", "系统配置",
)

_IRREVERSIBLE = (
    "delete", "remove", "drop", "destroy", "publish", "deploy", "transfer",
    "pay", "purchase", "checkout", "send", "submit", "uninstall",
    "删除", "发布", "部署", "支付", "下单", "转账", "发送", "提交", "卸载",
)

_READ = (
    "get", "list", "search", "read", "query", "fetch", "describe", "status",
    "snapshot", "screenshot", "health", "lookup", "查看", "查询", "读取", "截图",
)


def infer_risk(name: str, description: str = "") -> Risk:
    """Infer reversibility from the tool name: no allowlist, only a risk hint."""
    text = f"{name} {description}".lower()
    if any(k in text for k in _SENIOR):
        return Risk.SENIOR
    if any(k in text for k in _IRREVERSIBLE):
        return Risk.IRREVERSIBLE
    if any(k in text for k in _READ) or text.startswith(("get", "list", "read")):
        return Risk.READ
    return Risk.REVERSIBLE


def jsonable(obj: Any) -> Any:
    """Best-effort conversion of SDK objects into plain JSON-serialisable data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump())
        except Exception:  # noqa: BLE001 - defensive at the SDK boundary
            pass
    return str(obj)


def dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(jsonable(obj), ensure_ascii=False, **kwargs)
