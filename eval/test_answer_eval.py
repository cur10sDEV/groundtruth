import asyncio

import litellm
from answer_eval import evaluate_answer, judge_relevance, parse_relevance


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(_FakeMessage(content))]


def test_parse_relevance_json():
    assert parse_relevance('{"score": 0.9}') == 0.9
    assert parse_relevance("garbage") == 0.0


def test_evaluate_answer_scores_faithfulness_and_relevance(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _FakeResponse('{"score": 0.9}')

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    contexts = [
        {
            "id": "chunk-elec-refund",
            "text": "Electronics purchased within 30 days can be returned for a full refund.",
        }
    ]
    result = asyncio.run(
        evaluate_answer(
            "What is the refund policy for electronics?",
            "Electronics can be returned for a full refund within 30 days.",
            contexts,
        )
    )
    assert result == {"faithfulness": 0.9, "faithful": True, "relevance": 0.9}


def test_judge_relevance_fails_open_on_llm_error(monkeypatch):
    async def failing_acompletion(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(litellm, "acompletion", failing_acompletion)
    assert asyncio.run(judge_relevance("q", "a")) == 0.0


def test_judge_relevance_garbage_content_scores_zero(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _FakeResponse("garbage")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    assert asyncio.run(judge_relevance("q", "a")) == 0.0
