import json
import logging

from aio_pika import ExchangeType, connect_robust

from app.core.config import get_settings
from app.core.errors import IngestionError
from app.db import get_session
from app.models.document import Document, DocumentStatus
from app.rag.ingestion.pipeline import ingest_document

logger = logging.getLogger(__name__)
QUEUE = "ingestion"
DLQ = "ingestion.dlq"


async def declare_queues(channel) -> None:
    dlx = await channel.declare_exchange("ingestion.dlx", ExchangeType.DIRECT, durable=True)
    dlq = await channel.declare_queue(DLQ, durable=True, arguments={"x-queue-mode": "lazy"})
    await dlq.bind(dlx, routing_key=DLQ)
    await channel.declare_queue(
        QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "ingestion.dlx",
            "x-dead-letter-routing-key": DLQ,
            "x-message-ttl": 30000,
            "x-max-length": 10000,
        },
    )


declare = declare_queues


async def process_message(body: dict) -> None:
    doc_id = body.get("doc_id")
    s3_key = body.get("s3_key")
    if not doc_id or not s3_key:
        raise IngestionError(detail="message missing doc_id/s3_key")
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise IngestionError(detail=f"unknown doc {doc_id}")
        if doc.status == DocumentStatus.EMBEDDED:
            return
    await ingest_document(doc_id, s3_key)


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
                async with message.process(requeue=False):
                    body = json.loads(message.body)
                    await process_message(body)
            except Exception:
                logger.exception("ingestion failed, moving to DLQ", extra={"body": message.body})
