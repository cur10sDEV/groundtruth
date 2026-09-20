import asyncio

import litellm

from app.rag.retrieval.rewrite import (
    RewrittenQueries,
    fallback_rewritten,
    rewrite_query,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def test_fallback_preserves_query():
    r = fallback_rewritten("what is refund policy")
    assert r.canonical == "what is refund policy"
    assert r.queries == ["what is refund policy"]


def test_dataclass():
    r = RewrittenQueries(canonical="a", queries=["a", "b"])
    assert len(r.queries) == 2


def test_rewrite_query_parses_llm_json(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            _FakeMessage(
                '{"canonical": "what is the refund policy", '
                '"queries": ["how do refunds work", "refund policy explanation"]}'
            )
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = asyncio.run(rewrite_query("waht is refund polciy"))
    assert result.canonical == "what is the refund policy"
    assert result.queries == [
        "what is the refund policy",
        "how do refunds work",
        "refund policy explanation",
    ]
    assert captured["model"] == "openai/gpt-4o"
    assert captured["temperature"] == 0


def test_rewrite_query_never_breaks_retrieval_on_failure(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = asyncio.run(rewrite_query("anything"))
    assert result == fallback_rewritten("anything")
    assert result.canonical == "anything"
    assert result.queries == ["anything"]
