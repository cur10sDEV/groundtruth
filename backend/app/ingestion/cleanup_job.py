import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.qdrant_store import delete_points
from app.core.s3 import delete_prefix
from app.db import get_session
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.ingestion.pipeline import cleanup_stale
from app.rag.retrieval.cache import invalidate_for_doc

logger = logging.getLogger(__name__)


async def find_stale_doc_ids(limit: int = 100) -> list[str]:
    """Doc ids having ANY chunk older than the doc's current_version.

    A doc that already serves the new version but still holds old-version
    chunks (mixed window before cleanup) must be returned too.
    """
    async with get_session() as session:
        rows = await session.execute(
            select(Chunk.doc_id)
            .join(Document, Document.id == Chunk.doc_id)
            .where(Chunk.version < Document.current_version)
            .distinct()
            .limit(limit)
        )
        return [r.doc_id for r in rows]


async def reap_abandoned_pends() -> int:
    """Sweep PENDING docs whose upload never arrived; best-effort blob cleanup.

    The aware-UTC cutoff compares correctly on both backends: sqlite binds it
    as naive-UTC (matching its UTC CURRENT_TIMESTAMP) and postgres stores
    timestamptz.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=get_settings().reaper_pending_after_seconds)
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(Document).where(
                        Document.status == DocumentStatus.PENDING,
                        Document.created_at < cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
        for d in rows:
            try:
                delete_prefix(d.org_id, d.user_id, d.id)
            except Exception:
                logger.warning("reaper blob cleanup failed", extra={"doc_id": d.id})
            await session.delete(d)
        await session.commit()
        return len(rows)


async def complete_deletions() -> int:
    """Finish DELETING tombstones by mirroring the delete cascade, idempotently.

    The chunk sweep runs first: an in-flight ingest raced by deletion can
    commit chunk rows after the route's own chunk delete already ran, so the
    retry must re-run it. Every step is wrapped so a failure leaves the
    tombstone in place for the next interval.
    """
    completed = 0
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(Document).where(Document.status == DocumentStatus.DELETING)
                )
            )
            .scalars()
            .all()
        )
        for d in rows:
            try:
                await session.execute(delete(Chunk).where(Chunk.doc_id == d.id))
            except Exception:
                logger.warning("deletion chunk sweep failed; will retry", extra={"doc_id": d.id})
                continue
            try:
                delete_points([], {"must": [{"key": "doc_id", "match": {"value": d.id}}]})
            except Exception:
                logger.warning("deletion qdrant cleanup failed; will retry", extra={"doc_id": d.id})
                continue
            try:
                await invalidate_for_doc(d.org_id, d.id)
            except Exception:
                logger.warning(
                    "deletion cache invalidation failed; will retry", extra={"doc_id": d.id}
                )
                continue
            try:
                delete_prefix(d.org_id, d.user_id, d.id)
            except Exception:
                logger.warning("deletion blob cleanup failed; will retry", extra={"doc_id": d.id})
                continue
            await session.delete(d)
            completed += 1
        await session.commit()
        return completed


async def cleanup_job_loop(interval_seconds: int = 3600) -> None:
    while True:
        try:
            await complete_deletions()
            await reap_abandoned_pends()
            for doc_id in await find_stale_doc_ids():
                await cleanup_stale(doc_id)
        except Exception:
            logger.exception("cleanup job iteration failed")
        await asyncio.sleep(interval_seconds)
