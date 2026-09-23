from collections.abc import AsyncIterator

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db import get_sessionmaker
from app.models.chunk import Chunk
from app.rag.guardrails.rules import run_guardrails, validate_output
from app.rag.retrieval.cache import CachedEntry, get_cached, set_cached
from app.rag.retrieval.faithfulness import check_faithfulness
from app.rag.retrieval.filters import extract_filters
from app.rag.retrieval.generate import generate_answer, truncate_contexts
from app.rag.retrieval.retriever import get_retriever
from app.rag.retrieval.rewrite import rewrite_query

logger = get_logger(__name__)


async def resolve_text_for_chunk_ids(chunk_ids: list[str], org_id: str) -> list[dict]:
    if not chunk_ids:
        return []
    sm = get_sessionmaker()
    async with sm() as session:
        rows = (
            await session.execute(
                select(Chunk.id, Chunk.chunk_text).where(
                    Chunk.id.in_(chunk_ids), Chunk.org_id == org_id
                )
            )
        ).all()
    by_id = {str(row.id): row.chunk_text for row in rows}
    return [
        {"id": cid, "text": by_id[cid]}
        for cid in chunk_ids
        if cid in by_id  # drop rows that vanished (version flips) or fail RBAC
    ]


async def run_query(
    query: str, org_id: str, user_ids: list[str], feature_flags: dict, trace_id: str
) -> AsyncIterator[dict]:
    settings = get_settings()
    flags = feature_flags

    # guardrails: block injection/profanity, mask PII, clean input
    g = run_guardrails(query)
    cleaned = g.cleaned_text
    yield {
        "type": "status",
        "stage": "guardrails",
        "ok": g.passed,
        "reasons": g.reasons,
        "masked": g.masked,
    }
    if not g.passed:
        yield {
            "type": "done",
            "answer": "Query blocked by guardrails.",
            "chunk_ids": [],
            "doc_ids": [],
        }
        return

    # semantic cache — any cache failure (embedding/LLM/Redis) degrades to skip-cache
    cache_on = flags.get("cache.enabled", True)
    cached: CachedEntry | None = None
    if cache_on:
        try:
            cached = await get_cached(org_id, cleaned)
        except Exception as exc:
            logger.warning("cache lookup failed, proceeding without cache", extra={"exc": str(exc)})
        if cached is not None and cached.faithful:
            yield {"type": "status", "stage": "cache", "hit": True}
            yield {
                "type": "done",
                "answer": cached.answer,
                "chunk_ids": cached.chunk_ids,
                "doc_ids": cached.doc_ids,
            }
            return
        yield {"type": "status", "stage": "cache", "hit": False}

    # rewrite (use cleaned query)
    rewritten = await rewrite_query(cleaned)
    yield {"type": "status", "stage": "rewrite"}

    # filters
    filters = await extract_filters(cleaned)
    yield {"type": "status", "stage": "filters"}

    # retrieve
    retriever = get_retriever()
    all_chunks = []
    for q in rewritten.queries[:3]:
        all_chunks.extend(
            await retriever.retrieve(q, org_id, user_ids, filters.to_payload(), limit=5)
        )
    seen: set = set()
    chunk_ids: list[str] = []
    doc_by_chunk: dict[str, str] = {}
    for c in all_chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        chunk_ids.append(c.chunk_id)
        doc_by_chunk[c.chunk_id] = str(c.payload.get("doc_id", ""))
        if len(chunk_ids) >= 8:
            break

    # enrich from Postgres — Qdrant payload carries only chunk_text_hash, never real text
    resolved = await resolve_text_for_chunk_ids(chunk_ids[:8], org_id)
    contexts = [{"id": r["id"], "text": r["text"]} for r in resolved]

    # configurable context budget before generation
    contexts = truncate_contexts(contexts, settings.max_context_tokens)
    yield {"type": "status", "stage": "retrieve", "count": len(contexts)}

    if not contexts:
        yield {
            "type": "done",
            "answer": "I cannot confidently answer that based on the available documents.",
            "chunk_ids": [],
            "doc_ids": [],
        }
        return

    # generate (stream), capture model_used
    answer_parts = []
    model_used = settings.llm_primary_model
    async for ev in generate_answer(contexts, cleaned):
        if ev["type"] == "meta":
            model_used = ev["model_used"]
            yield {"type": "meta", "model_used": model_used}
        else:
            answer_parts.append(ev["text"])
            yield {"type": "token", "text": ev["text"]}
    answer = "".join(answer_parts)

    # faithfulness
    if flags.get("faithfulness.enabled", True):
        f = await check_faithfulness(cleaned, answer, contexts)
        yield {"type": "faithfulness", "faithful": f.faithful, "score": f.score}
        if not f.faithful:
            answer = "I cannot confidently answer that based on the available documents."
            yield {"type": "override", "answer": answer}

    # output security validation (mask PII, scan secret/harmful patterns)
    answer, warnings = validate_output(answer)
    if warnings:
        yield {"type": "output_warning", "warnings": warnings}

    chunk_ids = [c["id"] for c in contexts]
    doc_ids = list({doc_by_chunk[c["id"]] for c in contexts if c["id"] in doc_by_chunk})
    if cache_on:
        try:
            await set_cached(
                org_id,
                cleaned,
                CachedEntry(
                    answer=answer,
                    chunk_ids=chunk_ids,
                    doc_ids=doc_ids,
                    faithful=True,
                ),
            )
        except Exception as exc:
            logger.warning("cache write failed, proceeding without cache", extra={"exc": str(exc)})
    yield {"type": "done", "answer": answer, "chunk_ids": chunk_ids, "doc_ids": doc_ids}
