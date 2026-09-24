import json
import logging

from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage

from app.core.config import get_settings
from app.core.errors import IngestionError
from app.core.s3 import key_to_parts
from app.db import get_session
from app.models.document import Document, DocumentStatus
from app.rag.ingestion.pipeline import ingest_document, ingest_versioned

logger = logging.getLogger(__name__)
QUEUE = "ingestion"
DLQ = "ingestion.dlq"
RETRY = "ingestion.retry"
MAX_RETRIES = 3


async def declare_queues(channel: AbstractChannel) -> None:
    dlx = await channel.declare_exchange("ingestion.dlx", ExchangeType.DIRECT, durable=True)
    dlq = await channel.declare_queue(DLQ, durable=True, arguments={"x-queue-mode": "lazy"})
    await dlq.bind(dlx, routing_key=DLQ)
    await channel.declare_queue(
        QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "ingestion.dlx",
            "x-dead-letter-routing-key": DLQ,
            "x-max-length": 10000,
        },
    )
    await channel.declare_queue(
        RETRY,
        durable=True,
        arguments={
            "x-message-ttl": 30000,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": QUEUE,
        },
    )


declare = declare_queues


async def process_message(body: dict) -> None:
    doc_id = body.get("doc_id")
    if not doc_id:
        raise IngestionError(detail="message missing doc_id")
    if body.get("cancel"):
        logger.info("document cancellation noticed", extra={"doc_id": doc_id})
        return
    s3_key = body.get("s3_key")
    if not s3_key:
        raise IngestionError(detail="message missing s3_key")
    new_version = body.get("new_version")
    if new_version is not None and (
        isinstance(new_version, bool) or not isinstance(new_version, int) or new_version < 1
    ):
        raise IngestionError(detail=f"invalid new_version: {new_version}")
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise IngestionError(detail=f"unknown doc {doc_id}")
        # cross-tenant seam: the s3 key's org/user segments must match the
        # document owner or the message is poisoned → DLQ, never ingest
        try:
            key_org, key_user, _, _ = key_to_parts(s3_key)
        except ValueError as exc:
            raise IngestionError(detail=f"malformed s3 key: {s3_key}") from exc
        if key_org != doc.org_id or key_user != doc.user_id:
            raise IngestionError(detail=f"s3 key does not match document owner: {s3_key}")
        if new_version is not None:
            # Versioned re-ingest is also the recovery path for a previous FAILED
            # attempt, so a FAILED doc proceeds. Skip only when the flip already
            # happened (at-least-once redelivery after success).
            if doc.current_version >= new_version:
                return
        else:
            if doc.status == DocumentStatus.EMBEDDED:
                return
            if doc.status == DocumentStatus.FAILED:
                logger.info("skipping cancelled document", extra={"doc_id": doc_id})
                return
    if new_version is not None:
        await ingest_versioned(doc_id, s3_key, new_version)
    else:
        await ingest_document(doc_id, s3_key)


def _retry_count(message: AbstractIncomingMessage) -> int:
    deaths = (message.headers or {}).get("x-death") or []
    for entry in deaths:
        if entry.get("queue") == RETRY:
            return int(entry.get("count", 0))
    return 0


async def consume_loop() -> None:
    settings = get_settings()
    connection = await connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=4)
    await declare_queues(channel)
    queue = await channel.get_queue(QUEUE)
    async with queue.iterator() as qiter:
        async for message in qiter:
            try:
                body = json.loads(message.body)
                await process_message(body)
                await message.ack()
            except Exception:
                retries = _retry_count(message)
                if retries >= MAX_RETRIES:
                    logger.exception(
                        "ingestion failed after retries, moving to DLQ",
                        extra={"body": message.body, "retries": retries},
                    )
                    await message.reject(requeue=False)
                else:
                    logger.exception(
                        "ingestion failed, scheduling retry",
                        extra={"body": message.body, "retries": retries},
                    )
                    await channel.default_exchange.publish(
                        Message(
                            body=message.body,
                            headers=dict(message.headers or {}),
                            delivery_mode=DeliveryMode.PERSISTENT,
                        ),
                        routing_key=RETRY,
                    )
                    await message.ack()
