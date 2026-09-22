from __future__ import annotations

import fnmatch
import json

import pytest

import app.core.redis_store as redis_store
import app.rag.retrieval.cache as cache_module
from app.rag.retrieval.cache import (
    CachedEntry,
    get_cached,
    invalidate_for_doc,
    set_cached,
)

ORG = "org-1"
ORG_B = "org-2"

# stubbed dense_embed vectors (all unit tests; never hits the LLM)
VECTORS = {
    "what is rbac": [1.0, 0.0],
    # cos vs "what is rbac" ≈ 0.9899 (≥ 0.95 threshold)
    "what is rbac?": [0.98, 0.14],
    # unit vector; cos vs "what is rbac" ≈ 0.9806 (≥ 0.95, < 1.0)
    "what exactly is rbac": [0.98058, 0.196116],
    "what exactly is rbac?": [0.98058, 0.196116],
    # cos vs "what is rbac" ≈ 0.90 (< 0.95 threshold)
    "explain rbac": [0.9, 0.43589],
    # orthogonal to the rbac vectors
    "how do i reset my password": [0.0, 1.0],
}


def test_cached_entry_dataclass():
    e = CachedEntry(answer="a", chunk_ids=["c1"], doc_ids=["d1"], faithful=True)
    assert e.answer == "a"
    assert e.doc_ids == ["d1"]


class FakeRedis:
    def __init__(self):
        self.store: dict[str, dict] = {}
        self.sets: dict[str, set[str]] = {}
        self.ttls: dict[str, int | None] = {}

    async def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        self.store[key] = json.loads(json.dumps(value))
        self.ttls[key] = ttl

    async def get(self, key: str) -> dict | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)
        self.sets.pop(key, None)

    async def sadd(self, key: str, *members: str) -> None:
        self.sets.setdefault(key, set()).update(members)

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    async def keys_by_pattern(self, pattern: str) -> list[str]:
        return [k for k in sorted(self.store) if fnmatch.fnmatchcase(k, pattern)]


@pytest.fixture(autouse=True)
def _fresh_redis_singleton():
    """Reset the real client singleton so integration tests never reuse a dead event loop."""
    redis_store._cache = None
    yield
    redis_store._cache = None


@pytest.fixture
def embed_calls(monkeypatch):
    calls: list[list[str]] = []

    async def fake_dense_embed(texts):
        calls.append(list(texts))
        return [list(VECTORS[" ".join(t.lower().split())]) for t in texts]

    monkeypatch.setattr(cache_module, "dense_embed", fake_dense_embed)
    return calls


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache_module, "get_cache", lambda: fake)
    return fake


def _entry(answer: str, doc_ids=("doc-9",), faithful: bool = True) -> CachedEntry:
    return CachedEntry(answer=answer, chunk_ids=["c1"], doc_ids=list(doc_ids), faithful=faithful)


async def test_set_cached_returns_org_scoped_normalized_key(fake_redis, embed_calls):
    k1 = await set_cached(ORG, "  What is RBAC?  ", _entry("a1"))
    assert k1.startswith(f"cache:{ORG}:")

    k2 = await set_cached(ORG, "what is rbac?", _entry("a2"))
    assert k2 == k1  # case/whitespace normalization dedups onto one cache key


async def test_set_cached_stores_payload_embedding_ttl_and_reverse_index(fake_redis, embed_calls):
    key = await set_cached(ORG, "what is rbac", _entry("a1", doc_ids=("d1", "d2")))

    raw = fake_redis.store[key]
    assert raw["answer"] == "a1"
    assert raw["doc_ids"] == ["d1", "d2"]
    assert raw["faithful"] is True
    assert raw["embedding"] == VECTORS["what is rbac"]
    assert fake_redis.ttls[key] == 86400

    assert fake_redis.sets[f"cacheidx:{ORG}:d1"] == {key}
    assert fake_redis.sets[f"cacheidx:{ORG}:d2"] == {key}


async def test_get_cached_returns_none_without_embedding_when_cache_empty(fake_redis, embed_calls):
    assert await get_cached(ORG, "what is rbac") is None
    assert embed_calls == []  # no LLM spend when the org has nothing cached


async def test_get_cached_hits_for_semantically_similar_query(fake_redis, embed_calls):
    await set_cached(ORG, "what is rbac", _entry("a1"))

    hit = await get_cached(ORG, "What is RBAC?")  # cos ≈ 0.99 ≥ threshold

    assert hit == _entry("a1")


async def test_get_cached_misses_below_similarity_threshold(fake_redis, embed_calls):
    await set_cached(ORG, "what is rbac", _entry("a1"))

    assert await get_cached(ORG, "explain rbac") is None  # cos ≈ 0.90 < 0.95


async def test_get_cached_returns_closest_match_above_threshold(fake_redis, embed_calls):
    await set_cached(ORG, "what is rbac", _entry("a1"))
    await set_cached(ORG, "what exactly is rbac", _entry("a2"))

    hit = await get_cached(ORG, "what exactly is rbac?")

    assert hit is not None
    assert hit.answer == "a2"  # not the first-qualifying entry ("a1" sorts earlier, cos 0.98)


async def test_get_cached_is_org_scoped(fake_redis, embed_calls):
    await set_cached(ORG, "what is rbac", _entry("org-a answer"))
    await set_cached(ORG_B, "what is rbac", _entry("org-b answer"))

    hit_a = await get_cached(ORG, "what is rbac")
    hit_b = await get_cached(ORG_B, "what is rbac")

    assert hit_a is not None and hit_a.answer == "org-a answer"
    assert hit_b is not None and hit_b.answer == "org-b answer"


async def test_get_cached_skips_unusable_entries(fake_redis, embed_calls):
    key = await set_cached(ORG, "what is rbac", _entry("a1"))

    fake_redis.store[key].pop("embedding")  # legacy entry without an embedding
    assert await get_cached(ORG, "what is rbac") is None

    fake_redis.store[key]["embedding"] = [1.0, 0.0, 0.0]  # wrong dimensionality
    assert await get_cached(ORG, "what is rbac") is None


async def test_invalidate_for_doc_deletes_referencing_entries(fake_redis, embed_calls):
    k1 = await set_cached(ORG, "what is rbac", _entry("a1", doc_ids=("d1",)))
    k2 = await set_cached(ORG, "how do i reset my password", _entry("a2", doc_ids=("d2",)))

    removed = await invalidate_for_doc(ORG, "d1")

    assert removed == 1
    assert k1 not in fake_redis.store
    assert f"cacheidx:{ORG}:d1" not in fake_redis.sets
    assert await get_cached(ORG, "what is rbac") is None

    assert k2 in fake_redis.store
    hit = await get_cached(ORG, "how do i reset my password")
    assert hit is not None and hit.answer == "a2"


async def test_invalidate_for_doc_is_org_scoped(fake_redis, embed_calls):
    await set_cached(ORG, "what is rbac", _entry("a1", doc_ids=("d1",)))
    await set_cached(ORG_B, "what is rbac", _entry("b1", doc_ids=("d1",)))

    removed = await invalidate_for_doc(ORG, "d1")

    assert removed == 1
    hit = await get_cached(ORG_B, "what is rbac")
    assert hit is not None and hit.answer == "b1"  # other org untouched


async def test_invalidate_for_doc_returns_zero_when_nothing_indexed(fake_redis, embed_calls):
    assert await invalidate_for_doc(ORG, "never-seen") == 0


@pytest.mark.integration
async def test_live_semantic_cache_roundtrip_and_invalidation(monkeypatch):
    org, doc = "org-cache-live", "doc-cache-live"

    async def fake_dense_embed(texts):
        return [list(VECTORS[" ".join(t.lower().split())]) for t in texts]

    monkeypatch.setattr(cache_module, "dense_embed", fake_dense_embed)

    key = await set_cached(
        org,
        "what is rbac",
        CachedEntry(answer="live-a", chunk_ids=["c1"], doc_ids=[doc], faithful=True),
    )
    try:
        hit = await get_cached(org, "What is RBAC?")
        assert hit is not None and hit.answer == "live-a"

        assert await get_cached(org, "explain rbac") is None  # below threshold
        assert await get_cached(f"{org}-b", "what is rbac") is None  # other org, no leak

        assert await invalidate_for_doc(org, doc) == 1
        assert await get_cached(org, "what is rbac") is None
    finally:
        from app.core.redis_store import get_cache as _get_cache

        cache = _get_cache()
        await cache.delete(key)
        await cache.delete(f"cacheidx:{org}:{doc}")
