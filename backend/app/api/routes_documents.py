import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import delete, select, update

from app.auth.dependencies import get_current_user
from app.core.errors import ValidationError
from app.core.qdrant_store import delete_points
from app.core.s3 import delete_prefix, presign_upload
from app.db import get_session
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.retrieval.cache import invalidate_for_doc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


class SignIn(BaseModel):
    filename: str


@router.post("/sign")
async def sign_upload(body: SignIn, user: dict = Depends(get_current_user)) -> dict:
    filename = body.filename or "untitled"
    if "/" in filename or "\\" in filename:
        raise ValidationError(detail="invalid filename")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    async with get_session() as session:
        # same-filename re-upload for an already-EMBEDDED doc → new version of that doc
        versioned = (
            (
                await session.execute(
                    select(Document)
                    .where(
                        Document.org_id == user["org_id"],
                        Document.user_id == user["user_id"],
                        Document.original_filename == filename,
                        Document.status == DocumentStatus.EMBEDDED,
                    )
                    .order_by(Document.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        if versioned is not None:
            new_version = versioned.current_version + 1
            versioned.status = DocumentStatus.PENDING
            versioned.pending_version = new_version
            await session.commit()
            presigned = presign_upload(user["org_id"], user["user_id"], versioned.id, ext)
            return {
                "doc_id": versioned.id,
                "status": "pending",
                "new_version": new_version,
                "upload": {"url": presigned["url"], "fields": presigned["fields"]},
            }

        doc = Document(
            user_id=user["user_id"],
            org_id=user["org_id"],
            original_filename=filename,
            status=DocumentStatus.PENDING,
            content_hash="",  # unknown until the worker fetches the object
            current_version=1,
        )
        session.add(doc)
        await session.commit()
        presigned = presign_upload(user["org_id"], user["user_id"], doc.id, ext)
        return {
            "doc_id": doc.id,
            "status": "pending",
            "upload": {"url": presigned["url"], "fields": presigned["fields"]},
        }


@router.get("")
async def list_documents(user: dict = Depends(get_current_user)) -> list[dict]:
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(Document).where(
                        Document.org_id == user["org_id"],
                        Document.user_id == user["user_id"],
                        Document.status != DocumentStatus.DELETING,
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
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
        await session.execute(
            update(Document)
            .where(Document.id == doc_id, Document.org_id == user["org_id"])
            .values(status=DocumentStatus.DELETING, failure_reason=None)
        )
        await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        await session.commit()
    done = True
    try:
        delete_points(
            [],
            {"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
        )
    except Exception:
        logger.warning("delete: qdrant cleanup failed", extra={"doc_id": doc_id})
        done = False
    try:
        await invalidate_for_doc(user["org_id"], doc_id)
    except Exception:
        logger.warning("delete: cache invalidation failed", extra={"doc_id": doc_id})
        done = False
    try:
        delete_prefix(doc.org_id, doc.user_id, doc_id)
    except Exception:
        logger.warning("delete: blob cleanup failed", extra={"doc_id": doc_id})
        done = False
    if done:
        async with get_session() as session:
            d = await session.get(Document, doc_id)
            if d is not None:
                await session.delete(d)
                await session.commit()
    return {"doc_id": doc_id, "status": "deleted" if done else "deleting"}


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
