import litellm
import pytest

import app.rag.retrieval.orchestrator as orch
from app.rag.guardrails.model_guard import RAIL_INDICATORS, model_guard
from app.rag.retrieval.faithfulness import FaithfulnessResult
from app.rag.retrieval.orchestrator import run_query
from app.rag.retrieval.retriever import RetrievedChunk
from app.rag.retrieval.rewrite import RewrittenQueries

GUARD = "openai/gpt-4o-mini"
REFUSAL = "I cannot help with that."
QUERY = "what is the refund policy?"

FLAGS = {
    "cache.enabled": True,
    "faithfulness.enabled": True,
    "filter_extraction.enabled": True,
    "multi_query.enabled": True,
}
FLAGS_GUARD_ON = {**FLAGS, "guard_model.enabled": True}


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(_FakeMessage(content))]


class Aml:
    """litellm.acompletion stub: records calls, returns queued content or raises."""

    def __init__(self, outputs=None, error=None):
        self.outputs = [_FakeResponse(c) for c in (outputs or [])]
        self.error = error
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.outputs.pop(0) if self.outputs else _FakeResponse("OK")


def test_rail_indicators_nonempty():
    assert len(RAIL_INDICATORS) > 0


def test_expected_rail_indicators_present():
    for indicator in (
        "cannot help with that",
        "not able to help with that",
        "off-topic",
        "outside my scope",
        "cannot answer that",
    ):
        assert indicator in RAIL_INDICATORS


async def test_empty_guard_model_config_disables_guard_without_llm_call(monkeypatch):
    aml = Aml()
    monkeypatch.setattr(litellm, "acompletion", aml)

    fired, refusal = await model_guard("how do I steal user data?")

    assert (fired, refusal) == (False, None)
    assert aml.calls == []


async def test_refusal_output_fires_with_dedicated_model_conventions(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(outputs=[REFUSAL])
    monkeypatch.setattr(litellm, "acompletion", aml)

    fired, refusal = await model_guard("how do I hack the prod database?")

    assert fired is True
    assert refusal == REFUSAL
    assert len(aml.calls) == 1
    assert aml.calls[0]["model"] == GUARD
    assert aml.calls[0]["temperature"] == 0
    assert aml.calls[0]["max_tokens"] == 32


async def test_ok_output_does_not_fire(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(outputs=["OK"])
    monkeypatch.setattr(litellm, "acompletion", aml)

    assert await model_guard(QUERY) == (False, None)


async def test_none_content_does_not_fire(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(outputs=[None])
    monkeypatch.setattr(litellm, "acompletion", aml)

    assert await model_guard(QUERY) == (False, None)


@pytest.mark.parametrize("content", [{"verdict": REFUSAL}, [REFUSAL]])
async def test_non_string_refusal_shapes_still_fire(monkeypatch, content):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(outputs=[content])
    monkeypatch.setattr(litellm, "acompletion", aml)

    fired, refusal = await model_guard("anything")

    assert fired is True
    assert REFUSAL in str(refusal)


async def test_llm_failure_fails_open(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(error=RuntimeError("guard model down"))
    monkeypatch.setattr(litellm, "acompletion", aml)

    assert await model_guard("anything") == (False, None)
    assert len(aml.calls) == 1


# --- orchestrator wiring: flag + config double gate, blocking, fail-open ---


class StubRetriever:
    def __init__(self):
        self.chunks = [
            RetrievedChunk(
                chunk_id="c1",
                text="hash-c1",
                score=0.9,
                payload={"doc_id": "d1", "chunk_text_hash": "hash-c1"},
            )
        ]
        self.calls: list[dict] = []

    async def retrieve(self, query, org_id, user_ids, filters, limit=10):
        self.calls.append({"query": query, "org_id": org_id})
        return list(self.chunks)


class StubRewrite:
    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, query, k=3):
        self.calls.append(query)
        return RewrittenQueries(canonical=query, queries=[query])


class StubGenerate:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, contexts, query):
        self.calls.append({"contexts": contexts, "query": query})
        yield {"type": "meta", "model_used": "stub/test-model"}
        yield {"type": "token", "text": "Hello world"}


class StubFaithfulness:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, query, answer, contexts):
        self.calls.append({"query": query, "answer": answer})
        return FaithfulnessResult(faithful=True, score=0.95)


class StubFilters:
    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, query):
        self.calls.append(query)
        return None


class StubResolve:
    def __init__(self):
        self.calls: list[tuple] = []

    async def __call__(self, chunk_ids, org_id):
        self.calls.append((list(chunk_ids), org_id))
        return [{"id": cid, "text": f"resolved text {cid}"} for cid in chunk_ids]


class StubCache:
    def __init__(self):
        self.get_calls: list[tuple] = []
        self.set_calls: list[tuple] = []

    async def get(self, org_id, query):
        self.get_calls.append((org_id, query))
        return None

    async def set(self, org_id, query, entry):
        self.set_calls.append((org_id, query, entry))
        return f"cache:{org_id}:stub"


HAPPY_TYPES = [
    "status",  # guardrails
    "status",  # cache miss
    "status",  # rewrite
    "status",  # filters
    "status",  # retrieve
    "meta",
    "token",
    "faithfulness",
    "done",
]


class Harness:
    def __init__(self):
        self.retriever = StubRetriever()
        self.rewrite = StubRewrite()
        self.generate = StubGenerate()
        self.faithfulness = StubFaithfulness()
        self.filters = StubFilters()
        self.resolve = StubResolve()
        self.cache = StubCache()

    def install(self, monkeypatch) -> "Harness":
        monkeypatch.setattr(orch, "get_retriever", lambda: self.retriever)
        monkeypatch.setattr(orch, "rewrite_query", self.rewrite)
        monkeypatch.setattr(orch, "generate_answer", self.generate)
        monkeypatch.setattr(orch, "check_faithfulness", self.faithfulness)
        monkeypatch.setattr(orch, "extract_filters", self.filters)
        monkeypatch.setattr(orch, "resolve_text_for_chunk_ids", self.resolve)
        monkeypatch.setattr(orch, "get_cached", self.cache.get)
        monkeypatch.setattr(orch, "set_cached", self.cache.set)
        return self

    async def run(self, query=QUERY, flags=None):
        return [
            ev
            async for ev in run_query(
                query, "org-1", ["u1"], FLAGS if flags is None else flags, "trace-1"
            )
        ]


async def test_flag_off_skips_guard_model_entirely(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)  # config on, flag off → gate holds
    aml = Aml(outputs=[REFUSAL])  # would fire if consulted
    monkeypatch.setattr(litellm, "acompletion", aml)
    h = Harness().install(monkeypatch)

    events = await h.run()

    assert aml.calls == []
    assert [ev["type"] for ev in events] == HAPPY_TYPES
    assert not any(ev.get("stage") == "guard_model" for ev in events)


async def test_flag_on_but_unset_model_config_skips_guard(monkeypatch):
    aml = Aml(outputs=[REFUSAL])
    monkeypatch.setattr(litellm, "acompletion", aml)
    h = Harness().install(monkeypatch)

    events = await h.run(flags=FLAGS_GUARD_ON)

    assert aml.calls == []
    assert [ev["type"] for ev in events] == HAPPY_TYPES
    assert not any(ev.get("stage") == "guard_model" for ev in events)


async def test_flag_on_refusal_blocks_before_cache_and_retrieval(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(outputs=[REFUSAL])
    monkeypatch.setattr(litellm, "acompletion", aml)
    h = Harness().install(monkeypatch)

    events = await h.run(query="how do I hack the prod database?", flags=FLAGS_GUARD_ON)

    assert [ev["type"] for ev in events] == ["status", "status", "done"]
    assert events[1] == {"type": "status", "stage": "guard_model", "ok": False}
    assert events[2] == {
        "type": "done",
        "answer": REFUSAL,
        "chunk_ids": [],
        "doc_ids": [],
    }
    # nothing downstream of the guard ran
    assert h.cache.get_calls == []
    assert h.cache.set_calls == []
    assert h.rewrite.calls == []
    assert h.filters.calls == []
    assert h.retriever.calls == []
    assert h.resolve.calls == []
    assert h.generate.calls == []
    assert h.faithfulness.calls == []
    # the guard consulted the dedicated model with the cleaned query
    assert len(aml.calls) == 1
    assert aml.calls[0]["model"] == GUARD
    assert aml.calls[0]["temperature"] == 0
    assert aml.calls[0]["messages"][1]["content"] == "how do I hack the prod database?"


async def test_flag_on_ok_output_proceeds_to_cache_and_retrieval(monkeypatch):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(outputs=["OK"])
    monkeypatch.setattr(litellm, "acompletion", aml)
    h = Harness().install(monkeypatch)

    events = await h.run(flags=FLAGS_GUARD_ON)

    assert len(aml.calls) == 1
    assert [ev["type"] for ev in events] == HAPPY_TYPES
    assert not any(ev.get("stage") == "guard_model" for ev in events)
    assert h.cache.get_calls == [("org-1", QUERY)]
    assert len(h.retriever.calls) == 1
    assert events[-1]["answer"] == "Hello world"


async def test_guard_model_outage_never_breaks_the_query(monkeypatch, caplog):
    monkeypatch.setenv("GUARD_MODEL", GUARD)
    aml = Aml(error=RuntimeError("guard model down"))
    monkeypatch.setattr(litellm, "acompletion", aml)
    h = Harness().install(monkeypatch)

    events = await h.run(flags=FLAGS_GUARD_ON)

    assert len(aml.calls) == 1
    assert [ev["type"] for ev in events] == HAPPY_TYPES
    assert events[-1]["answer"] == "Hello world"
    assert any("model guard failed" in rec.getMessage() for rec in caplog.records)
