import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import delete, select, update

from app.auth.dependencies import get_current_user
from app.core.errors import ValidationError
from app.core.s3 import presign_upload
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
            result = await session.execute(
                update(Document)
                .where(Document.id == versioned.id, Document.status == DocumentStatus.EMBEDDED)
                .values(status=DocumentStatus.PENDING, pending_version=new_version)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            if result.rowcount == 0:
                # the EMBEDDED row vanished mid-sign (deleted, etc.) — fall
                # through to the fresh-document branch instead of stomping
                versioned = None
            else:
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
                "failure_reason": d.failure_reason,
            }
            for d in rows
        ]


@router.get("/{doc_id}")
async def get_document(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"] or doc.user_id != user["user_id"]:
            raise ValidationError(detail="document not found")
        return {
            "id": doc.id,
            "filename": doc.original_filename,
            "status": doc.status.value,
            "version": doc.current_version,
            "failure_reason": doc.failure_reason,
        }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"] or doc.user_id != user["user_id"]:
            raise ValidationError(detail="document not found")
        await session.execute(
            update(Document)
            .where(Document.id == doc_id, Document.org_id == user["org_id"])
            .values(status=DocumentStatus.DELETING, failure_reason=None)
        )
        await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        await session.commit()
    cache_cleared = True
    try:
        await invalidate_for_doc(user["org_id"], doc_id)
    except Exception:
        logger.warning("delete: cache invalidation failed", extra={"doc_id": doc_id}, exc_info=True)
        cache_cleared = False
    response = {"doc_id": doc_id, "status": "deleted"}
    if not cache_cleared:
        response["note"] = "changes may take a few minutes to fully take effect"
    return response


@router.get("/{doc_id}/chunks")
async def get_doc_chunks(doc_id: str, user: dict = Depends(get_current_user)) -> list[dict]:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"] or doc.user_id != user["user_id"]:
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
