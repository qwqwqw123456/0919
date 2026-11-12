from __future__ import annotations

from typing import Any

from ..types import Risk, SourceKind, ToolSpec

_DESKTOP_TOOLS: list[tuple[str, str, dict[str, Any], Risk]] = [
    ("desktop.screenshot", "截取整个桌面屏幕，返回图片路径",
     {"type": "object", "properties": {}}, Risk.READ),
    ("desktop.screen_size", "获取屏幕分辨率",
     {"type": "object", "properties": {}}, Risk.READ),
    ("desktop.windows", "列出当前可见窗口（标题 / 位置 / 是否前台）",
     {"type": "object", "properties": {}}, Risk.READ),
    ("desktop.focus", "将指定标题的窗口置于前台",
     {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}, Risk.REVERSIBLE),
    ("desktop.move", "移动鼠标到坐标",
     {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
      "required": ["x", "y"]}, Risk.REVERSIBLE),
    ("desktop.click", "在坐标处点击（覆盖非 Web 的 GUI 程序）",
     {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                                      "button": {"type": "string", "enum": ["left", "right", "middle"],
                                                 "default": "left"},
                                      "clicks": {"type": "integer", "default": 1}},
      "required": ["x", "y"]}, Risk.REVERSIBLE),
    ("desktop.type_text", "输入文本（需要已聚焦目标输入框）",
     {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, Risk.REVERSIBLE),
    ("desktop.hotkey", "发送组合键，如 ctrl+c / alt+tab",
     {"type": "object", "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
      "required": ["keys"]}, Risk.REVERSIBLE),
]


def desktop_specs(source: str = "native") -> list[ToolSpec]:
    return [
        ToolSpec(name=n, description=d, input_schema=s, kind=SourceKind.DESKTOP,
                 risk=r, source=source, idempotent=r is Risk.READ)
        for n, d, s, r in _DESKTOP_TOOLS
    ]
