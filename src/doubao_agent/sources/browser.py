from __future__ import annotations

from typing import Any

from ..types import Risk, SourceKind, ToolSpec

_BROWSER_TOOLS: list[tuple[str, str, dict[str, Any], Risk]] = [
    ("browser.open", "在浏览器中打开 URL（豆包原生浏览器）",
     {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}, Risk.REVERSIBLE),
    ("browser.snapshot", "获取当前页面的可理解快照（DOM + 可见文本 + 可交互元素）",
     {"type": "object", "properties": {"max_chars": {"type": "integer", "default": 8000}}}, Risk.READ),
    ("browser.screenshot", "对当前页面截图，返回图片路径",
     {"type": "object", "properties": {"full_page": {"type": "boolean", "default": False}}}, Risk.READ),
    ("browser.click", "点击元素（CSS 选择器或可见文本）",
     {"type": "object", "properties": {"selector": {"type": "string"}, "text": {"type": "string"}}}, Risk.REVERSIBLE),
    ("browser.type", "在输入框中输入文本，可选回车提交",
     {"type": "object", "properties": {"selector": {"type": "string"}, "text": {"type": "string"},
                                      "submit": {"type": "boolean", "default": False}},
      "required": ["text"]}, Risk.REVERSIBLE),
    ("browser.upload", "向文件输入框上传云电脑上的文件",
     {"type": "object", "properties": {"selector": {"type": "string"}, "path": {"type": "string"}},
      "required": ["path"]}, Risk.REVERSIBLE),
    ("browser.tabs", "标签页管理：list / new / switch / close",
     {"type": "object", "properties": {"action": {"type": "string", "enum": ["list", "new", "switch", "close"]},
                                      "target": {"type": "string"}, "url": {"type": "string"}},
      "required": ["action"]}, Risk.REVERSIBLE),
    ("browser.extract", "按选择器或整页提取结构化文本/表格",
     {"type": "object", "properties": {"selector": {"type": "string"}}}, Risk.READ),
    ("browser.back", "浏览器后退",
     {"type": "object", "properties": {}}, Risk.REVERSIBLE),
]


def browser_specs(source: str = "native") -> list[ToolSpec]:
    return [
        ToolSpec(name=n, description=d, input_schema=s, kind=SourceKind.BROWSER,
                 risk=r, source=source, idempotent=r is Risk.READ)
        for n, d, s, r in _BROWSER_TOOLS
    ]
