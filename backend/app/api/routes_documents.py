import hashlib
import logging

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import select

from app.auth.dependencies import get_current_user
from app.core.errors import ValidationError
from app.core.s3 import put_object
from app.db import get_session
from app.ingestion.publisher import publish_ingestion
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.retrieval.cache import invalidate_for_doc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/upload")
async def upload_document(
    user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError(detail="file exceeds 50MB limit")
    content_hash = hashlib.sha256(data).hexdigest()
    async with get_session() as session:
        existing = (
            (
                await session.execute(
                    select(Document).where(
                        Document.content_hash == content_hash,
                        Document.org_id == user["org_id"],
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing and existing.status == DocumentStatus.EMBEDDED:
            return {"doc_id": existing.id, "status": "duplicate", "already_embedded": True}

        doc = Document(
            user_id=user["user_id"],
            org_id=user["org_id"],
            original_filename=file.filename or "untitled",
            status=DocumentStatus.PENDING,
            content_hash=content_hash,
            current_version=1,
        )
        session.add(doc)
        await session.commit()

        s3_key = put_object(
            user["org_id"], user["user_id"], doc.id, file.filename or "untitled", data
        )
        await publish_ingestion(doc.id, s3_key)
        return {"doc_id": doc.id, "status": "processing", "s3_key": s3_key}


@router.get("")
async def list_documents(user: dict = Depends(get_current_user)) -> list[dict]:
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(Document).where(
                        Document.org_id == user["org_id"],
                        Document.user_id == user["user_id"],
                    )
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "id": d.id,
                "filename": d.original_filename,
                "status": d.status.value,
                "version": d.current_version,
            }
            for d in rows
        ]


@router.get("/{doc_id}")
async def get_document(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
        return {
            "id": doc.id,
            "filename": doc.original_filename,
            "status": doc.status.value,
            "version": doc.current_version,
        }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    from app.ingestion.cancel import cancel_document

    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
    await cancel_document(doc_id)
    await invalidate_for_doc(user["org_id"], doc_id)
    return {"doc_id": doc_id, "status": "cancelled"}


@router.get("/{doc_id}/chunks")
async def get_doc_chunks(doc_id: str, user: dict = Depends(get_current_user)) -> list[dict]:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
        rows = (
            (
                await session.execute(
                    select(Chunk).where(
                        Chunk.doc_id == doc_id,
                        Chunk.org_id == user["org_id"],
                        Chunk.version == doc.current_version,
                    )
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "id": c.id,
                "page": c.page_number,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
            }
            for c in rows
        ]
