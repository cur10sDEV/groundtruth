from collections.abc import AsyncIterator

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import ensure_trace, trace_step
from app.db import get_sessionmaker
from app.models.chunk import Chunk
from app.rag.guardrails.rules import run_guardrails, validate_output
from app.rag.retrieval.cache import CachedEntry, get_cached, set_cached
from app.rag.retrieval.faithfulness import check_faithfulness
from app.rag.retrieval.filters import extract_filters
from app.rag.retrieval.generate import generate_answer, truncate_contexts
from app.rag.retrieval.rerank import RerankerDisabledError, rerank
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
    root = ensure_trace(trace_id)
    try:
        # guardrails: block injection/profanity, mask PII, clean input
        with trace_step("guardrails", trace_id, parent=root) as span:
            g = run_guardrails(query)
            cleaned = g.cleaned_text
            span.update(
                input={"query": query},
                output={"passed": g.passed, "reasons": g.reasons, "masked": g.masked},
            )
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

        # optional model-based guard gate (feature-flagged)
        if flags.get("guard_model.enabled", False) and settings.guard_model:
            from app.rag.guardrails.model_guard import model_guard

            with trace_step("guard_model", trace_id, parent=root) as span:
                span.update(input={"query": cleaned})
                fired, refusal = await model_guard(cleaned)
                span.update(output={"fired": fired, "refusal": refusal})
                if fired:
                    yield {"type": "status", "stage": "guard_model", "ok": False}
                    yield {
                        "type": "done",
                        "answer": refusal or "Query blocked.",
                        "chunk_ids": [],
                        "doc_ids": [],
                    }
                    return

        # semantic cache — any cache failure (embedding/LLM/Redis) degrades to skip-cache
        cache_on = flags.get("cache.enabled", True)
        cached: CachedEntry | None = None
        if cache_on:
            with trace_step("cache", trace_id, parent=root) as span:
                span.update(input={"org_id": org_id, "query": cleaned})
                try:
                    cached = await get_cached(org_id, cleaned)
                except Exception as exc:
                    logger.warning(
                        "cache lookup failed, proceeding without cache",
                        extra={"exc": str(exc)},
                    )
                    span.update(output={"hit": False, "error": str(exc)})
                else:
                    span.update(output={"hit": cached is not None and cached.faithful})
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
        with trace_step("rewrite", trace_id, parent=root) as span:
            span.update(input={"query": cleaned})
            rewritten = await rewrite_query(cleaned)
            span.update(
                output={"canonical": rewritten.canonical, "queries": list(rewritten.queries)}
            )
            yield {"type": "status", "stage": "rewrite"}

        # filters (flag-gated: off → no LLM filter extraction, empty payload downstream)
        with trace_step("filters", trace_id, parent=root) as span:
            span.update(input={"query": cleaned})
            if flags.get("filter_extraction.enabled", True):
                filters = await extract_filters(cleaned)
            else:
                filters = None
            span.update(
                output={"payload": filters.to_payload() if filters else None},
                metadata={"enabled": flags.get("filter_extraction.enabled", True)},
            )
            yield {"type": "status", "stage": "filters"}

        # retrieve (multi-query flag decides whether to expand beyond the canonical query)
        with trace_step("retrieve", trace_id, parent=root) as span:
            retriever = get_retriever()
            queries = (
                rewritten.queries[:3]
                if flags.get("multi_query.enabled", True)
                else [rewritten.canonical]
            )
            filter_payload = filters.to_payload() if filters else {}
            span.update(
                input={
                    "queries": queries,
                    "org_id": org_id,
                    "user_ids": user_ids,
                    "filters": filter_payload,
                }
            )
            all_chunks = []
            for q in queries:
                all_chunks.extend(
                    await retriever.retrieve(q, org_id, user_ids, filter_payload, limit=5)
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
            resolved = await resolve_text_for_chunk_ids(chunk_ids, org_id)
            # drop empty/whitespace-only chunk texts before the budget so all-empty docs
            # fall through to the empty-contexts refusal instead of reaching generation
            contexts = [{"id": r["id"], "text": r["text"]} for r in resolved if r["text"].strip()]

            # configurable context budget before generation
            contexts = truncate_contexts(contexts, settings.max_context_tokens)
            span.update(output={"chunk_ids": chunk_ids, "count": len(contexts)})
            yield {"type": "status", "stage": "retrieve", "count": len(contexts)}

        if not contexts:
            yield {
                "type": "done",
                "answer": "I cannot confidently answer that based on the available documents.",
                "chunk_ids": [],
                "doc_ids": [],
            }
            return

        # reranker gate — scores the real (enriched) texts, so it runs after enrichment
        # + truncation, then re-truncates because the reranked order/top_n may change the
        # budget fit. Fail-open per spec: flagged on but not configured or the reranker
        # errors → keep original order, never break the query.
        if flags.get("reranker.enabled", False):
            with trace_step("rerank", trace_id, parent=root) as span:
                span.update(
                    input={"query": cleaned, "top_n": 5, "contexts": [c["id"] for c in contexts]}
                )
                try:
                    contexts = await rerank(cleaned, contexts, top_n=5)
                    contexts = truncate_contexts(contexts, settings.max_context_tokens)
                except RerankerDisabledError:
                    span.update(output={"reranked": False, "reason": "disabled"})  # fail open
                except Exception as exc:
                    logger.warning("rerank failed, keeping original order", extra={"exc": str(exc)})
                    span.update(output={"reranked": False, "error": str(exc)})
                else:
                    span.update(output={"reranked": True, "contexts": [c["id"] for c in contexts]})

        # generate (stream), capture model_used
        with trace_step("generate", trace_id, parent=root) as span:
            span.update(input={"query": cleaned, "contexts": [c["id"] for c in contexts]})
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
            span.update(output={"answer": answer, "model_used": model_used})

        # faithfulness
        if flags.get("faithfulness.enabled", True):
            with trace_step("faithfulness", trace_id, parent=root) as span:
                span.update(input={"query": cleaned, "answer": answer})
                f = await check_faithfulness(cleaned, answer, contexts)
                span.update(output={"faithful": f.faithful, "score": f.score})
                yield {"type": "faithfulness", "faithful": f.faithful, "score": f.score}
                if not f.faithful:
                    answer = "I cannot confidently answer that based on the available documents."
                    yield {"type": "override", "answer": answer}

        # output security validation (mask PII, scan secret/harmful patterns)
        with trace_step("output_validation", trace_id, parent=root) as span:
            span.update(input={"answer": answer})
            answer, warnings = validate_output(answer)
            span.update(output={"answer": answer, "warnings": warnings})
            if warnings:
                yield {"type": "output_warning", "warnings": warnings}

        chunk_ids = [c["id"] for c in contexts]
        # insertion-ordered dedup keeps the SSE payload deterministic
        doc_ids = list(
            dict.fromkeys(doc_by_chunk[c["id"]] for c in contexts if c["id"] in doc_by_chunk)
        )
        if cache_on:
            with trace_step("cache", trace_id, parent=root) as span:
                span.update(
                    input={
                        "org_id": org_id,
                        "query": cleaned,
                        "chunk_ids": chunk_ids,
                        "doc_ids": doc_ids,
                    }
                )
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
                    logger.warning(
                        "cache write failed, proceeding without cache", extra={"exc": str(exc)}
                    )
                    span.update(output={"written": False, "error": str(exc)})
                else:
                    span.update(output={"written": True})
        yield {"type": "done", "answer": answer, "chunk_ids": chunk_ids, "doc_ids": doc_ids}
    finally:
        root.end()
