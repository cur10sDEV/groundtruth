import logging

from sqlalchemy import delete, select

from app.core.qdrant_store import delete_points
from app.db import get_session
from app.ingestion.publisher import publish_message
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


async def cancel_document(doc_id: str) -> None:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return
        doc.status = DocumentStatus.FAILED
        partial_ids = list(
            (await session.execute(select(Chunk.id).where(Chunk.doc_id == doc_id))).scalars().all()
        )
        if partial_ids:
            delete_points(
                partial_ids,
                {"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
            )
        await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        await session.commit()
    await publish_message({"doc_id": doc_id, "cancel": True})
