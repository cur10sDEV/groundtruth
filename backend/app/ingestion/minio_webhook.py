import logging
from urllib.parse import unquote

from fastapi import APIRouter, FastAPI, Request

from app.core.errors import IngestionError
from app.ingestion.publisher import publish_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/internal/minio-event")
async def minio_event(request: Request) -> dict:
    data = await request.json()
    for record in data.get("Records", []):
        if "ObjectRemoved" in record.get("eventName", ""):
            continue
        key = unquote(record.get("s3", {}).get("object", {}).get("key", ""))
        if not key:
            continue
        parts = key.split("/")
        # key: documents/<org>/<user>/<doc>/<uuid>.<ext>
        if len(parts) < 5:
            raise IngestionError(detail=f"unexpected s3 key: {key}")
        doc_id = parts[3]
        await publish_ingestion(doc_id, key)
    return {"status": "ok"}


def register_minio_webhook(app: FastAPI) -> None:
    app.include_router(router)
