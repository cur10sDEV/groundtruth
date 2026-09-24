import json
import logging

from aio_pika import DeliveryMode, Message, connect_robust

from app.core.config import get_settings
from app.ingestion.consumer import QUEUE, declare

logger = logging.getLogger(__name__)


async def publish_message(payload: dict, routing_key: str = QUEUE) -> None:
    settings = get_settings()
    connection = await connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        await declare(channel)  # declares ingestion, ingestion.retry (TTL dead-letters back), DLQ
        await channel.default_exchange.publish(
            Message(
                body=json.dumps(payload).encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=routing_key,
        )
    finally:
        await connection.close()


async def publish_ingestion(doc_id: str, s3_key: str, new_version: int | None = None) -> None:
    payload: dict = {"doc_id": doc_id, "s3_key": s3_key}
    if new_version is not None:
        payload["new_version"] = new_version
    await publish_message(payload)
