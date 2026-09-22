import asyncio

import litellm
import pytest

from app.core.errors import LLMError
from app.rag.retrieval.faithfulness import FaithfulnessResult, check_faithfulness
from app.rag.retrieval.generate import (
    build_context_block,
    generate_answer,
    truncate_contexts,
)


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content, model="openai/gpt-4o"):
        self.model = model
        self.choices = [_FakeStreamChoice(content)]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def _fake_stream(chunks):
    async def fake_acompletion(**kwargs):
        async def gen():
            for c in chunks:
                yield c

        return gen()

    return fake_acompletion


def test_build_context_block_numbers_citations():
    block = build_context_block([{"text": "Alpha"}, {"text": "Beta"}])
    assert "[1]" in block and "[2]" in block
    assert "Alpha" in block and "Beta" in block


def test_truncate_contexts_respects_budget():
    ctx = [{"text": "x" * 40}, {"text": "y" * 40}]
    out = truncate_contexts(ctx, max_tokens=12)
    # 40 chars ~ 11 tokens each (chars/4 + 1); budget 12 fits only the first
    assert len(out) == 1
    assert out[0]["text"].startswith("x")


def test_faithfulness_dataclass():
    r = FaithfulnessResult(faithful=True, score=0.9)
    assert r.faithful and r.score == 0.9


def test_generate_answer_yields_meta_then_tokens(monkeypatch):
    chunks = [
        _FakeChunk("Hello", model="openai/gpt-4o-mini"),
        _FakeChunk(" world"),
    ]
    monkeypatch.setattr(litellm, "acompletion", _fake_stream(chunks))

    async def collect():
        return [e async for e in generate_answer([{"text": "Alpha"}], "hi")]

    events = asyncio.run(collect())
    assert events[0] == {"type": "meta", "model_used": "openai/gpt-4o-mini"}
    assert events[1:] == [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " world"},
    ]


def test_generate_answer_prompt_requires_citations_and_refusal(monkeypatch):
    captured = {}
    inner = _fake_stream([])

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return await inner(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def collect():
        return [e async for e in generate_answer([{"text": "Alpha"}], "hi")]

    events = asyncio.run(collect())
    system = captured["messages"][0]["content"]
    assert "[n]" in system
    assert "I cannot confidently answer that based on the available documents." in system
    assert captured["stream"] is True
    assert captured["max_tokens"] == 2048
    assert captured["model"] == "openai/gpt-4o"
    # empty stream still yields meta with the configured primary model
    assert events == [{"type": "meta", "model_used": "openai/gpt-4o"}]


def test_generate_answer_raises_llm_error_on_failure(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def collect():
        return [e async for e in generate_answer([{"text": "Alpha"}], "hi")]

    with pytest.raises(LLMError):
        asyncio.run(collect())


def test_check_faithfulness_scores_groundedness(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(_FakeMessage('{"score": 0.9}'))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    result = asyncio.run(check_faithfulness("q", "a", [{"text": "c"}]))
    assert result == FaithfulnessResult(faithful=True, score=0.9)
    assert captured["model"] == "openai/gpt-4o"
    assert captured["temperature"] == 0


def test_check_faithfulness_threshold_is_07(monkeypatch):
    async def low(**kwargs):
        return _FakeResponse(_FakeMessage('{"score": 0.69}'))

    monkeypatch.setattr(litellm, "acompletion", low)
    assert asyncio.run(check_faithfulness("q", "a", [{"text": "c"}])).faithful is False

    async def boundary(**kwargs):
        return _FakeResponse(_FakeMessage('{"score": 0.7}'))

    monkeypatch.setattr(litellm, "acompletion", boundary)
    assert asyncio.run(check_faithfulness("q", "a", [{"text": "c"}])).faithful is True


def test_check_faithfulness_wraps_llm_failures(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    with pytest.raises(LLMError):
        asyncio.run(check_faithfulness("q", "a", [{"text": "c"}]))
