from __future__ import annotations


def test_host_provider_normalizes_shapes() -> None:
    from doubao_agent.providers.host import HostProvider

    assert HostProvider._normalize("你好") == {"content": "你好", "tool_calls": []}

    simple = HostProvider._normalize(
        {"content": "hi", "tool_calls": [{"id": "a", "name": "ping", "arguments": {}}]}
    )
    assert simple["content"] == "hi"
    assert simple["tool_calls"][0]["name"] == "ping"

    openai_style = HostProvider._normalize(
        {
            "choices": [
                {
                    "message": {
                        "content": "x",
                        "tool_calls": [
                            {"id": "b", "function": {"name": "go", "arguments": '{"n": 1}'}}
                        ],
                    }
                }
            ]
        }
    )
    assert openai_style["content"] == "x"
    assert openai_style["tool_calls"][0]["name"] == "go"
    assert openai_style["tool_calls"][0]["arguments"] == {"n": 1}


async def test_host_provider_completes_via_bridge() -> None:
    from doubao_agent.providers.host import HostProvider

    class FakeBridge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def invoke(self, capability: str, payload: dict) -> dict:
            self.calls.append((capability, payload))
            return {"content": "pong", "tool_calls": []}

        async def list_capabilities(self) -> list:
            return []

        async def aclose(self) -> None:
            return None

    bridge = FakeBridge()
    provider = HostProvider(bridge, capability="model.complete", model="doubao")
    out = await provider.complete([{"role": "user", "content": "hi"}], [])

    assert out["content"] == "pong"
    capability, payload = bridge.calls[0]
    assert capability == "model.complete"
    assert payload["model"] == "doubao"
    assert payload["messages"][0]["content"] == "hi"


def test_build_provider_host_branch() -> None:
    from doubao_agent.providers import build_provider
    from doubao_agent.providers.host import HostProvider
    from doubao_agent.providers.mock import MockProvider

    class Dummy:
        pass

    provider = build_provider(
        {"provider": "host", "capability": "model.complete", "model": "x"}, bridge=Dummy()
    )
    assert isinstance(provider, HostProvider)

    # 宿主桥缺失时退回 mock，避免运行时崩溃
    fallback = build_provider({"provider": "host"}, bridge=None)
    assert isinstance(fallback, MockProvider)


async def test_openai_compat_parses_tool_calls() -> None:
    import httpx

    from doubao_agent.providers.openai_compat import OpenAICompatProvider

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "knowledge.search", "arguments": '{"query":"x"}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            },
        )

    client = httpx.AsyncClient(
        base_url="https://llm.example.com/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer sk-test"},
    )
    provider = OpenAICompatProvider(
        base_url="https://llm.example.com/v1",
        model="my-model",
        client=client,
    )
    out = await provider.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])

    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "my-model"
    assert seen["body"]["tool_choice"] == "auto"
    assert out["content"] == "ok"
    assert out["tool_calls"][0]["name"] == "knowledge.search"
    assert out["tool_calls"][0]["arguments"] == {"query": "x"}
    assert out["usage"]["total_tokens"] == 8
    await client.aclose()


def test_openai_without_base_url_falls_back_to_mock() -> None:
    from doubao_agent.providers import build_provider
    from doubao_agent.providers.mock import MockProvider

    assert isinstance(build_provider({"provider": "openai"}), MockProvider)
    assert isinstance(
        build_provider({"provider": "openai", "base_url": "   "}), MockProvider
    )
