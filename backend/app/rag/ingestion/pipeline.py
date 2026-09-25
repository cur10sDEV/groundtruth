import asyncio
import hashlib
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.core.metrics import INGESTION_FAILED, INGESTION_PROCESSED
from app.core.qdrant_store import delete_points, upsert_points_batch
from app.core.s3 import get_object
from app.db import get_session
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.chunkers.text import get_chunker
from app.rag.embed.embeddings import dense_embed, sparse_embed
from app.rag.ingestion.parsers import parse_bytes
from app.rag.retrieval.cache import invalidate_for_doc

logger = get_logger(__name__)
EMBED_BATCH = 32
NAMESPACE_RAG = uuid.uuid5(uuid.NAMESPACE_DNS, "rag-prod.chunks")


def _derive_chunk_id(doc_id: str, version: int, chunk_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE_RAG, f"{doc_id}:{version}:{chunk_index}"))


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_persistable(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not set(stripped) <= {"#"}


def _cleanup_partial_qdrant(doc_id: str, version: int) -> None:
    try:
        delete_points(
            [],
            {
                "must": [
                    {"key": "doc_id", "match": {"value": doc_id}},
                    {"key": "version", "match": {"value": version}},
                ]
            },
        )
    except Exception:
        logger.warning(
            "failed to delete orphaned qdrant points",
            extra={"doc_id": doc_id, "version": version},
            exc_info=True,
        )


async def _reset_version(doc_id: str, version: int, session: AsyncSession) -> None:
    """Clean slate for an ingest attempt: drop any leftovers from a previous attempt."""
    await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id, Chunk.version == version))
    _cleanup_partial_qdrant(doc_id, version)


async def _finalize(doc_id: str, content_hash: str, new_version: int | None = None) -> None:
    """Guarded completion: flip to EMBEDDED (and roll the version) only while PROCESSING."""
    values: dict[str, object] = {
        "status": DocumentStatus.EMBEDDED,
        "pending_version": None,
        "content_hash": content_hash,
    }
    if new_version is not None:
        values["current_version"] = new_version
    async with get_session() as session:
        result = await session.execute(
            update(Document)
            .where(Document.id == doc_id, Document.status == DocumentStatus.PROCESSING)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    if result.rowcount == 0:
        raise IngestionError(detail=f"document finalization lost the race: {doc_id}")


async def _mark_failed(doc_id: str, reason: str | None = None) -> bool:
    """Guarded failure transition; a lost race (e.g. to DELETING) is fine and not an error."""
    async with get_session() as session:
        result = await session.execute(
            update(Document)
            .where(Document.id == doc_id, Document.status == DocumentStatus.PROCESSING)
            .values(
                status=DocumentStatus.FAILED,
                pending_version=None,
                failure_reason=reason,
            )
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    if result.rowcount == 0:
        logger.info(
            "failure mark lost the race; concurrent state change won",
            extra={"doc_id": doc_id},
        )
        return False
    return True


async def _embed_and_store(doc: Document, chunks: list[dict], upserted: list[str]) -> None:
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        texts = [c["chunk_text"] for c in batch]
        dense = await dense_embed(texts)
        sparse = sparse_embed(texts)
        points = []
        for c, d, s in zip(batch, dense, sparse, strict=True):
            points.append(
                {
                    "id": c["id"],
                    "dense": d,
                    "sparse_indices": s.indices,
                    "sparse_values": s.values,
                    "payload": {
                        "doc_id": doc.id,
                        "user_id": doc.user_id,
                        "org_id": doc.org_id,
                        "chunk_text_hash": hashlib.sha256(c["chunk_text"].encode()).hexdigest(),
                        "version": c["version"],
                        "page_number": c["page_number"],
                        "chunk_index": c["chunk_index"],
                        "word_count": len(c["chunk_text"].split()),
                        "char_count": len(c["chunk_text"]),
                        "type": doc.original_filename.rsplit(".", 1)[-1].lower(),
                    },
                }
            )
        upsert_points_batch(points)
        upserted.extend(p["id"] for p in points)


async def _ingest(doc_id: str, s3_key: str, version: int | None = None) -> int:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise IngestionError(detail=f"document not found: {doc_id}")
        chunk_version = doc.current_version if version is None else version
        dedup = version is None
        claimed = await session.execute(
            update(Document)
            .where(
                Document.id == doc_id,
                Document.status.in_(
                    (
                        DocumentStatus.PENDING,
                        DocumentStatus.PROCESSING,
                        DocumentStatus.FAILED,
                    )
                ),
            )
            .values(status=DocumentStatus.PROCESSING, pending_version=chunk_version)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        if claimed.rowcount == 0:
            raise IngestionError(detail=f"document not processing-claimable: {doc_id}")
        upserted: list[str] = []
        try:
            raw = get_object(s3_key)
            content_hash = _hash(raw)
            if dedup and content_hash == doc.content_hash:
                await _finalize(doc_id, content_hash=doc.content_hash)
                INGESTION_PROCESSED.inc()
                return 0

            await _reset_version(doc_id, chunk_version, session)

            try:
                parsed = parse_bytes(doc.original_filename, raw)
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(detail=f"parse failed for {doc_id}: {exc}") from exc

            chunker = get_chunker(doc.original_filename.rsplit(".", 1)[-1].lower())
            chunks: list[dict] = []
            for page in parsed.pages:
                for cd in chunker.chunk(page.text, page_number=page.page_number):
                    if not _is_persistable(cd.text):
                        continue
                    chunks.append(
                        {
                            "id": _derive_chunk_id(doc.id, chunk_version, len(chunks)),
                            "doc_id": doc.id,
                            "user_id": doc.user_id,
                            "org_id": doc.org_id,
                            "chunk_text": cd.text,
                            "page_number": cd.page_number,
                            "start_offset": cd.start_offset,
                            "end_offset": cd.end_offset,
                            "version": chunk_version,
                            "chunk_index": cd.chunk_index,
                        }
                    )

            await _embed_and_store(doc, chunks, upserted)

            session.add_all(
                Chunk(
                    **{
                        k: c[k]
                        for k in (
                            "id",
                            "doc_id",
                            "user_id",
                            "org_id",
                            "chunk_text",
                            "page_number",
                            "start_offset",
                            "end_offset",
                            "version",
                        )
                    }
                )
                for c in chunks
            )
            await session.commit()
            await _finalize(doc_id, content_hash, new_version=version)
            INGESTION_PROCESSED.inc()
            return len(chunks)
        except asyncio.CancelledError:
            if upserted:
                _cleanup_partial_qdrant(doc_id, chunk_version)
            await session.rollback()
            if await _mark_failed(doc_id):
                INGESTION_FAILED.inc()
            raise
        except Exception:
            if upserted:
                _cleanup_partial_qdrant(doc_id, chunk_version)
            await session.rollback()
            if await _mark_failed(doc_id):
                INGESTION_FAILED.inc()
            raise


async def ingest_document(doc_id: str, s3_key: str) -> int:
    return await _ingest(doc_id, s3_key)


async def ingest_versioned(doc_id: str, s3_key: str, new_version: int) -> int:
    count = await _ingest(doc_id, s3_key, version=new_version)
    await _invalidate_cache(doc_id)
    return count


async def _invalidate_cache(doc_id: str) -> None:
    try:
        async with get_session() as session:
            doc = await session.get(Document, doc_id)
            if doc is None:
                return
            await invalidate_for_doc(doc.org_id, doc_id)
    except Exception:
        logger.warning("cache invalidation failed", extra={"doc_id": doc_id})


async def cleanup_stale(doc_id: str) -> int:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return 0
        stale = list(
            (
                await session.execute(
                    select(Chunk.id).where(
                        Chunk.doc_id == doc_id, Chunk.version < doc.current_version
                    )
                )
            )
            .scalars()
            .all()
        )
        if stale:
            delete_points(
                stale,
                {"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
            )
        result = await session.execute(
            select(Chunk).where(Chunk.doc_id == doc_id, Chunk.version < doc.current_version)
        )
        for c in result.scalars():
            await session.delete(c)
        await session.commit()
        return len(stale)
