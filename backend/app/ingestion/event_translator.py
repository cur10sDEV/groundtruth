import json
import logging
from urllib.parse import unquote

from aio_pika import ExchangeType, connect_robust

from app.core.config import get_settings
from app.core.metrics import EVENTS_DROPPED
from app.core.s3 import key_to_parts
from app.db import get_session
from app.ingestion.publisher import publish_ingestion
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)
EVENTS_QUEUE = "minio.events"
EVENTS_EXCHANGE = "minio.events"
FORWARDABLE = {DocumentStatus.PENDING, DocumentStatus.PROCESSING}


def _drop(reason: str, key: str = "") -> bool:
    EVENTS_DROPPED.labels(reason=reason).inc()
    logger.warning("event dropped", extra={"reason": reason, "key": key})
    return False


def _record(event: dict) -> dict:
    # MinIO notification payloads nest the S3 record under Records[0]
    # (with a lowercase eventName and a percent-encoded object key);
    # the flat top-level shape is accepted for symmetry with the unit tests.
    records = event.get("Records")
    if isinstance(records, list) and records:
        return records[0]
    return event


async def translate_event(event: dict) -> bool:
    record = _record(event)
    name = str(record.get("eventName") or record.get("EventName") or "")
    if not name.startswith("s3:ObjectCreated:"):
        return _drop("not_created", name)
    s3 = record.get("s3") or {}
    if (s3.get("bucket") or {}).get("name") != get_settings().s3_bucket:
        return _drop("wrong_bucket")
    key = unquote(str((s3.get("object") or {}).get("key", "")))
    try:
        key_org, key_user, key_doc, _ = key_to_parts(key)
    except ValueError:
        return _drop("malformed_key", key)
    async with get_session() as session:
        doc = await session.get(Document, key_doc)
        if doc is None:
            return _drop("missing_row", key)
        if doc.status not in FORWARDABLE:
            return _drop("not_pending", key)
        if doc.org_id != key_org or doc.user_id != key_user:
            return _drop("ownership_mismatch", key)
        new_version = doc.pending_version
    await publish_ingestion(key_doc, key, new_version=new_version)
    return True


async def translate_loop() -> None:
    settings = get_settings()
    connection = await connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    exchange = await channel.declare_exchange(EVENTS_EXCHANGE, ExchangeType.DIRECT, durable=True)
    queue = await channel.declare_queue(EVENTS_QUEUE, durable=True)
    await queue.bind(exchange, routing_key=EVENTS_QUEUE)
    async with queue.iterator() as qiter:
        async for message in qiter:
            try:
                await translate_event(json.loads(message.body))
            except Exception:
                logger.exception("translate failed", extra={"body": message.body})
            finally:
                await message.ack()  # noise never retries — drops are terminal
