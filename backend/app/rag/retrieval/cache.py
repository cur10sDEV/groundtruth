import hashlib
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.redis_store import get_cache
from app.rag.embed.embeddings import dense_embed

# Spec §Redis: "similarity-threshold matching" — cosine similarity required for a cache hit.
SIMILARITY_THRESHOLD = 0.95


@dataclass
class CachedEntry:
    answer: str
    chunk_ids: list[str]
    doc_ids: list[str]
    faithful: bool


def _key(org_id: str, query: str) -> str:
    norm = " ".join(query.lower().split())
    digest = hashlib.sha256(norm.encode()).hexdigest()[:32]
    return f"cache:{org_id}:{digest}"


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (norm_a * norm_b)


async def set_cached(org_id: str, query: str, entry: CachedEntry) -> str:
    key = _key(org_id, query)
    vec = (await dense_embed([query]))[0]
    cache = get_cache()
    await cache.set(
        key,
        {
            "answer": entry.answer,
            "chunk_ids": entry.chunk_ids,
            "doc_ids": entry.doc_ids,
            "faithful": entry.faithful,
            "embedding": vec,
        },
        ttl=get_settings().cache_ttl_seconds,
    )
    for doc_id in entry.doc_ids:
        await cache.sadd(f"cacheidx:{org_id}:{doc_id}", key)
    return key


async def get_cached(org_id: str, query: str) -> CachedEntry | None:
    cache = get_cache()
    keys = await cache.keys_by_pattern(f"cache:{org_id}:*")
    if not keys:
        return None
    query_vec = (await dense_embed([query]))[0]
    best_raw: dict | None = None
    best_sim = -1.0
    for k in keys:
        raw = await cache.get(k)
        if raw is None or not isinstance(raw.get("embedding"), list):
            continue
        sim = _cosine(query_vec, raw["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_raw = raw
    if best_raw is None or best_sim < SIMILARITY_THRESHOLD:
        return None
    return CachedEntry(
        answer=best_raw["answer"],
        chunk_ids=best_raw["chunk_ids"],
        doc_ids=best_raw["doc_ids"],
        faithful=best_raw["faithful"],
    )


async def invalidate_for_doc(org_id: str, doc_id: str) -> int:
    cache = get_cache()
    keys = await cache.smembers(f"cacheidx:{org_id}:{doc_id}")
    for k in keys:
        await cache.delete(k)
    await cache.delete(f"cacheidx:{org_id}:{doc_id}")
    return len(keys)
