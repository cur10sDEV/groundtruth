import logging

from sqlalchemy import delete, update

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
        pending = doc.pending_version
        result = await session.execute(
            update(Document)
            .where(
                Document.id == doc_id,
                Document.status.in_((DocumentStatus.PENDING, DocumentStatus.PROCESSING)),
            )
            .values(status=DocumentStatus.FAILED, pending_version=None)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 0:
            logger.info(
                "cancel lost the race; concurrent state change won",
                extra={"doc_id": doc_id},
            )
            return
        if pending is not None:
            delete_points(
                [],
                {
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}},
                        {"key": "version", "range": {"gte": pending}},
                    ]
                },
            )
            await session.execute(
                delete(Chunk).where(Chunk.doc_id == doc_id, Chunk.version >= pending)
            )
        await session.commit()
    await publish_message({"doc_id": doc_id, "cancel": True})
