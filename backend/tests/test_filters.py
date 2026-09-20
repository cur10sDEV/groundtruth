import asyncio

import litellm

from app.rag.retrieval.filters import Filters, default_filters, extract_filters


class _FakeFunction:
    def __init__(self, arguments: str):
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, arguments: str):
        self.function = _FakeFunction(arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def test_filters_to_payload_drops_none():
    f = Filters(year=2025, type=None, tags=["hr"], topic=None, page_number=None)
    assert f.to_payload() == {"year": 2025, "tags": ["hr"]}


def test_default_filters_all_none():
    f = default_filters()
    assert f.to_payload() == {}


def test_extract_filters_uses_tool_call_arguments(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        msg = _FakeMessage(
            content=None,
            tool_calls=[
                _FakeToolCall('{"year": 2025, "tags": ["hr"], "topic": "vacation policy"}')
            ],
        )
        return _FakeResponse(msg)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = asyncio.run(extract_filters("hr docs from 2025 about vacation policy"))
    assert result == Filters(year=2025, tags=["hr"], topic="vacation policy")
    assert captured["model"] == "openai/gpt-4o"
    assert captured["temperature"] == 0


def test_extract_filters_falls_back_to_message_content(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _FakeResponse(_FakeMessage(content='{"type": "policy"}', tool_calls=None))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = asyncio.run(extract_filters("policy documents"))
    assert result == Filters(type="policy")


def test_extract_filters_never_blocks_retrieval_on_failure(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = asyncio.run(extract_filters("anything"))
    assert result == default_filters()
    assert result.to_payload() == {}
