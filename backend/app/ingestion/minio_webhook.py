import logging
from urllib.parse import unquote

from fastapi import APIRouter, FastAPI, Request

from app.core.config import get_settings
from app.core.errors import AuthenticationError, IngestionError
from app.core.logging import new_correlation_id, set_correlation_id
from app.ingestion.publisher import publish_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()

WEBHOOK_SECRET_HEADER = "X-Webhook-Secret"


@router.post("/internal/minio-event")
async def minio_event(request: Request) -> dict:
    set_correlation_id(new_correlation_id())
    settings = get_settings()
    provided = request.headers.get(WEBHOOK_SECRET_HEADER)
    # Unset secret disables the endpoint entirely (direct publish is canonical);
    # a configured secret must be presented and match exactly.
    if not settings.webhook_secret or provided != settings.webhook_secret:
        raise AuthenticationError(detail="invalid or missing webhook secret")
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
