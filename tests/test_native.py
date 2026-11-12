from __future__ import annotations

from conftest import FakeBridge

from doubao_agent.registry import CapabilityRegistry
from doubao_agent.types import Risk, ToolCall


async def test_native_source_discovers_every_capability() -> None:
    from doubao_agent.sources.native import NativeSource

    bridge = FakeBridge(
        [
            {"name": "clipboard.read", "description": "读取剪贴板"},
            {"name": "notify.send", "description": "发送系统通知",
             "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                              "required": ["text"]}},
            {"name": "media.camera", "description": "调用摄像头", "risk": "L2"},
        ]
    )
    source = NativeSource(bridge)
    specs = await source.list_tools()

    assert {spec.name for spec in specs} == {"clipboard.read", "notify.send", "media.camera"}
    camera = next(spec for spec in specs if spec.name == "media.camera")
    assert camera.risk is Risk.IRREVERSIBLE

    result = await source.call_tool(
        ToolCall(name="notify.send", arguments={"text": "hi"}, call_id="n1")
    )
    assert result.ok
    assert bridge.calls == [("notify.send", {"text": "hi"})]


async def test_native_source_falls_back_when_manifest_empty() -> None:
    from doubao_agent.sources.native import NativeSource

    source = NativeSource(FakeBridge([]))
    names = {spec.name for spec in await source.list_tools()}
    assert "browser.open" in names
    assert "desktop.screenshot" in names


async def test_native_tools_keep_their_namespace() -> None:
    from doubao_agent.sources.native import NativeSource

    registry = CapabilityRegistry()
    registry.add_source(NativeSource(FakeBridge([{"name": "notify.send", "description": "通知"}])))
    await registry.refresh()
    assert [spec.name for spec in registry.list_tools()] == ["notify.send"]
