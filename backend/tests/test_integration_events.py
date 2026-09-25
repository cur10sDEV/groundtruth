import asyncio
import json
import time
import uuid
from urllib.parse import unquote

import httpx
import pytest
from aio_pika import ExchangeType, connect_robust
from aio_pika.exceptions import QueueEmpty

from app.core.config import get_settings
from app.core.s3 import presign_upload

pytestmark = pytest.mark.integration


async def _events_queue(channel):
    exchange = await channel.declare_exchange("minio.events", ExchangeType.DIRECT, durable=True)
    queue = await channel.declare_queue("minio.events", durable=True)
    await queue.bind(exchange, routing_key="minio.events")
    return queue


async def test_presigned_upload_event_reaches_minio_events_queue():
    settings = get_settings()
    presigned = presign_upload("org-live", "user-live", str(uuid.uuid4()), "txt")
    key = presigned["key"]

    # mirror translate_loop's declarations so the event is routed to a queue
    # even with no worker running, and purge any leftovers from earlier runs
    connection = await connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        queue = await _events_queue(channel)
        await queue.purge()
    finally:
        await connection.close()

    resp = httpx.post(
        presigned["url"],
        data=presigned["fields"],
        files={"file": ("x.txt", b"live trigger bytes", "text/plain")},
    )
    assert resp.status_code == 204  # MinIO answers 204 No Content on POST success

    connection = await connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        queue = await channel.get_queue("minio.events")
        # basic_get is a single-shot RPC: poll until the event lands or bail
        deadline = time.monotonic() + 15.0
        message = None
        while message is None and time.monotonic() < deadline:
            try:
                message = await queue.get(fail=True, timeout=5.0)
            except QueueEmpty:
                await asyncio.sleep(0.2)
        assert message is not None, "no event arrived on minio.events within 15s"
        await message.ack()
    finally:
        await connection.close()

    event = json.loads(message.body)
    assert event["EventName"] == "s3:ObjectCreated:Post"
    record = next(r for r in event["Records"] if unquote(r["s3"]["object"]["key"]) == key)
    assert record["eventName"] == "s3:ObjectCreated:Post"
    assert record["s3"]["bucket"]["name"] == settings.s3_bucket
