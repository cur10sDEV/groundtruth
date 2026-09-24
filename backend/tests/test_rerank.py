import pytest

import app.rag.retrieval.orchestrator as orch
import app.rag.retrieval.rerank as rerank_module
from app.core.config import get_settings
from app.core.errors import DomainError
from app.rag.retrieval.rerank import RerankerDisabledError, rerank
from app.rag.retrieval.retriever import RetrievedChunk

QUERY = "what is the refund policy?"


async def _noop_log(*args, **kwargs):
    pass


# gates on/off per test; cache + faithfulness off to keep the harness minimal
BASE_FLAGS = {
    "cache.enabled": False,
    "faithfulness.enabled": False,
    "multi_query.enabled": True,
    "filter_extraction.enabled": True,
    "reranker.enabled": False,
}


def test_reranker_disabled_error():
    err = RerankerDisabledError()
    assert err.status_code == 503


def test_reranker_disabled_error_is_domain_error():
    assert issubclass(RerankerDisabledError, DomainError)


# --- rerank module (no live Cohere: cohere.Client is stubbed) ---


class _StubRerankResult:
    def __init__(self, index):
        self.index = index


class _StubRerankResponse:
    def __init__(self, indices):
        self.results = [_StubRerankResult(i) for i in indices]


class _StubCohereClient:
    def __init__(self, indices=None, raise_exc=None):
        self.indices = indices or []
        self.raise_exc = raise_exc
        self.calls = []

    def rerank(self, model, query, documents, top_n):
        self.calls.append(
            {"model": model, "query": query, "documents": list(documents), "top_n": top_n}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return _StubRerankResponse(self.indices)


async def test_rerank_without_api_key_raises_disabled(monkeypatch):
    monkeypatch.setenv("RERANKER_API_KEY", "")
    with pytest.raises(RerankerDisabledError):
        await rerank("q", [{"id": "c1", "text": "alpha"}])


async def test_rerank_unknown_provider_raises_disabled(monkeypatch):
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_PROVIDER", "other")
    stub = _StubCohereClient(indices=[0])
    monkeypatch.setattr(rerank_module.cohere, "Client", lambda key: stub)
    with pytest.raises(RerankerDisabledError):
        await rerank("q", [{"id": "c1", "text": "alpha"}])


async def test_rerank_reorders_contexts_via_cohere(monkeypatch):
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    monkeypatch.setenv("RERANKER_MODEL", "rerank-english-v3.0")
    contexts = [
        {"id": "c1", "text": "alpha"},
        {"id": "c2", "text": "beta"},
        {"id": "c3", "text": "gamma"},
    ]
    stub = _StubCohereClient(indices=[2, 0, 1])
    monkeypatch.setattr(rerank_module.cohere, "Client", lambda key: stub)

    out = await rerank("q", contexts, top_n=3)

    assert [c["id"] for c in out] == ["c3", "c1", "c2"]
    assert stub.calls == [
        {
            "model": "rerank-english-v3.0",
            "query": "q",
            "documents": ["alpha", "beta", "gamma"],
            "top_n": 3,
        }
    ]


async def test_rerank_transient_failure_fails_open(monkeypatch, caplog):
    monkeypatch.setenv("RERANKER_API_KEY", "test-key")
    contexts = [{"id": "c1", "text": "alpha"}]
    stub = _StubCohereClient(raise_exc=RuntimeError("cohere down"))
    monkeypatch.setattr(rerank_module.cohere, "Client", lambda key: stub)

    out = await rerank("q", contexts)

    assert out == contexts  # original order preserved
    assert any("rerank failed" in rec.getMessage() for rec in caplog.records)


# --- orchestrator flag gates (stubs wired at module boundaries) ---


def _chunk(cid: str, doc_id: str) -> RetrievedChunk:
    # payload carries only the hash placeholder, never real text
    return RetrievedChunk(
        chunk_id=cid,
        text=f"hash-{cid}",
        score=0.5,
        payload={"doc_id": doc_id, "chunk_text_hash": f"hash-{cid}"},
    )


class _StubRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls: list[dict] = []

    async def retrieve(self, query, org_id, user_ids, filters, limit=10):
        self.calls.append(
            {
                "query": query,
                "org_id": org_id,
                "user_ids": list(user_ids),
                "filters": filters,
                "limit": limit,
            }
        )
        return list(self.chunks)


class _StubGenerate:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, contexts, query):
        self.calls.append({"contexts": [dict(c) for c in contexts], "query": query})
        yield {"type": "meta", "model_used": "stub/test-model"}
        yield {"type": "token", "text": "Hello world"}


class _StubRewrite:
    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, query, k=3):
        from app.rag.retrieval.rewrite import RewrittenQueries

        self.calls.append(query)
        return RewrittenQueries(canonical="q-canonical", queries=["q-canonical", "q-paraphrase"])


class _StubFilters:
    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, query):
        from app.rag.retrieval.filters import Filters

        self.calls.append(query)
        return Filters(topic="refunds")


class _StubResolve:
    def __init__(self):
        self.calls: list[tuple] = []

    async def __call__(self, chunk_ids, org_id):
        self.calls.append((list(chunk_ids), org_id))
        return [{"id": cid, "text": f"resolved-text-{cid}"} for cid in chunk_ids]


class _StubCache:
    def __init__(self):
        self.get_calls: list[tuple] = []
        self.set_calls: list[tuple] = []

    async def get(self, org_id, query):
        self.get_calls.append((org_id, query))
        return None

    async def set(self, org_id, query, entry):
        self.set_calls.append((org_id, query, entry))


class _StubRerank:
    def __init__(self, output=None, raise_exc=None):
        self.output = output
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def __call__(self, query, contexts, top_n=5):
        self.calls.append({"query": query, "contexts": [dict(c) for c in contexts], "top_n": top_n})
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.output is not None:
            return [dict(c) for c in self.output]
        return [dict(c) for c in contexts]


class Harness:
    """Offline wiring of every module boundary run_query consumes."""

    def __init__(self):
        self.retriever = _StubRetriever()
        self.generate = _StubGenerate()
        self.rewrite = _StubRewrite()
        self.filters = _StubFilters()
        self.resolve = _StubResolve()
        self.cache = _StubCache()
        self.rerank = _StubRerank()

    def install(self, monkeypatch) -> "Harness":
        monkeypatch.setattr(orch, "get_retriever", lambda: self.retriever)
        monkeypatch.setattr(orch, "generate_answer", self.generate)
        monkeypatch.setattr(orch, "rewrite_query", self.rewrite)
        monkeypatch.setattr(orch, "extract_filters", self.filters)
        monkeypatch.setattr(orch, "resolve_text_for_chunk_ids", self.resolve)
        monkeypatch.setattr(orch, "get_cached", self.cache.get)
        monkeypatch.setattr(orch, "set_cached", self.cache.set)
        monkeypatch.setattr(orch, "rerank", self.rerank)
        monkeypatch.setattr(orch, "_log_query", _noop_log)
        return self

    async def run(self, flags):
        return [
            ev
            async for ev in orch.run_query(
                QUERY, "org-1", ["u1"], {**BASE_FLAGS, **flags}, trace_id="trace-1"
            )
        ]


async def test_multi_query_enabled_expands_queries(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1")]
    h.install(monkeypatch)

    await h.run({"multi_query.enabled": True})

    assert [c["query"] for c in h.retriever.calls] == ["q-canonical", "q-paraphrase"]


async def test_multi_query_disabled_uses_single_canonical_query(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1")]
    h.install(monkeypatch)

    await h.run({"multi_query.enabled": False})

    assert h.rewrite.calls == [QUERY]
    assert [c["query"] for c in h.retriever.calls] == ["q-canonical"]


async def test_filter_extraction_enabled_passes_payload(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1")]
    h.install(monkeypatch)

    await h.run({"filter_extraction.enabled": True})

    assert h.filters.calls == [QUERY]
    for c in h.retriever.calls:
        assert c["filters"] == {"topic": "refunds"}


async def test_filter_extraction_disabled_yields_empty_filter_payload(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1")]
    h.install(monkeypatch)

    await h.run({"filter_extraction.enabled": False})

    assert h.filters.calls == []
    for c in h.retriever.calls:
        assert c["filters"] == {}


async def test_reranker_flag_off_skips_rerank(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1")]
    h.install(monkeypatch)

    events = await h.run({"reranker.enabled": False})

    assert h.rerank.calls == []
    assert h.generate.calls[0]["contexts"] == [{"id": "c1", "text": "resolved-text-c1"}]
    assert events[-1]["type"] == "done"


async def test_reranker_flag_on_reranks_enriched_contexts_and_retruncates(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1"), _chunk("c2", "d2"), _chunk("c3", "d3")]
    h.rerank = _StubRerank(
        output=[
            {"id": "c3", "text": "resolved-text-c3"},
            {"id": "c1", "text": "resolved-text-c1"},
        ]
    )
    h.install(monkeypatch)

    real_truncate = orch.truncate_contexts
    truncate_calls: list[tuple] = []

    def _recording_truncate(contexts, max_tokens):
        truncate_calls.append(([c["id"] for c in contexts], max_tokens))
        return real_truncate(contexts, max_tokens)

    monkeypatch.setattr(orch, "truncate_contexts", _recording_truncate)

    events = await h.run({"reranker.enabled": True})

    # rerank scored the enriched (Postgres-resolved) contexts, never the hash payloads
    assert h.rerank.calls == [
        {
            "query": QUERY,
            "contexts": [
                {"id": "c1", "text": "resolved-text-c1"},
                {"id": "c2", "text": "resolved-text-c2"},
                {"id": "c3", "text": "resolved-text-c3"},
            ],
            "top_n": 5,
        }
    ]
    # re-truncated after rerank, on the reranked order, within the same budget
    assert truncate_calls == [
        (["c1", "c2", "c3"], get_settings().max_context_tokens),
        (["c3", "c1"], get_settings().max_context_tokens),
    ]
    # generation used the reranked, re-truncated contexts
    assert h.generate.calls[0]["contexts"] == [
        {"id": "c3", "text": "resolved-text-c3"},
        {"id": "c1", "text": "resolved-text-c1"},
    ]
    done = events[-1]
    assert done["type"] == "done"
    assert done["chunk_ids"] == ["c3", "c1"]
    assert done["doc_ids"] == ["d3", "d1"]


async def test_reranker_disabled_error_fails_open(monkeypatch):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1"), _chunk("c2", "d2")]
    h.rerank = _StubRerank(raise_exc=RerankerDisabledError())
    h.install(monkeypatch)

    events = await h.run({"reranker.enabled": True})

    assert h.rerank.calls  # the gate tried
    # fail-open: generation proceeds with the original enriched order, no crash
    assert h.generate.calls[0]["contexts"] == [
        {"id": "c1", "text": "resolved-text-c1"},
        {"id": "c2", "text": "resolved-text-c2"},
    ]
    assert events[-1] == {
        "type": "done",
        "answer": "Hello world",
        "chunk_ids": ["c1", "c2"],
        "doc_ids": ["d1", "d2"],
        "query_id": "trace-1",
    }


async def test_reranker_transient_error_fails_open(monkeypatch, caplog):
    h = Harness()
    h.retriever.chunks = [_chunk("c1", "d1")]
    h.rerank = _StubRerank(raise_exc=RuntimeError("cohere timeout"))
    h.install(monkeypatch)

    events = await h.run({"reranker.enabled": True})

    assert h.generate.calls[0]["contexts"] == [{"id": "c1", "text": "resolved-text-c1"}]
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "Hello world"
    assert any("rerank failed" in rec.getMessage() for rec in caplog.records)
