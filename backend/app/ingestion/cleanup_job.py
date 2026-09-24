import asyncio
import logging

from sqlalchemy import select

from app.db import get_session
from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.ingestion.pipeline import cleanup_stale

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


async def cleanup_job_loop(interval_seconds: int = 3600) -> None:
    while True:
        try:
            for doc_id in await find_stale_doc_ids():
                await cleanup_stale(doc_id)
        except Exception:
            logger.exception("cleanup job iteration failed")
        await asyncio.sleep(interval_seconds)
