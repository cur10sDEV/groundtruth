import asyncio
import json

import pytest
from aio_pika import Message
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

import app.db as db_module
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_correlation_id
from app.db import get_session, init_db
from app.ingestion import cancel as cancel_module
from app.ingestion import cleanup_job as cleanup_module
from app.ingestion import consumer, minio_webhook
from app.ingestion import publisher as publisher_module
from app.ingestion.publisher import publish_ingestion
from app.main import create_app
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus

DOC_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"
ORG_ID = "33333333-3333-3333-3333-333333333333"
S3_KEY = "documents/org/user/doc/sample.txt"
DOC_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
DOC_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"


def test_publish_ingestion_signature():
    assert callable(publish_ingestion)


@pytest.fixture
def _sqlite_database_url(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(db_module, "_engine", None)
    monkeypatch.setattr(db_module, "_sessionmaker", None)


@pytest.fixture
async def db(_sqlite_database_url):
    await init_db()
    yield
    if db_module._engine is not None:
        await db_module._engine.dispose()


async def _seed_doc(
    doc_id: str,
    current_version: int = 1,
    status: DocumentStatus = DocumentStatus.PENDING,
    pending_version: int | None = None,
) -> None:
    async with get_session() as session:
        session.add(
            Document(
                id=doc_id,
                user_id=USER_ID,
                org_id=ORG_ID,
                original_filename="sample.txt",
                status=status,
                content_hash="hash",
                current_version=current_version,
                pending_version=pending_version,
            )
        )
        await session.commit()


async def _seed_chunk(doc_id: str, version: int) -> str:
    async with get_session() as session:
        chunk = Chunk(
            doc_id=doc_id,
            user_id=USER_ID,
            org_id=ORG_ID,
            chunk_text="some text",
            page_number=0,
            start_offset=0,
            end_offset=len("some text"),
            version=version,
        )
        session.add(chunk)
        await session.commit()
        return chunk.id


async def _get_doc(doc_id: str) -> Document:
    async with get_session() as session:
        return await session.get(Document, doc_id)


async def _chunk_ids(doc_id: str) -> set[str]:
    async with get_session() as session:
        rows = await session.execute(select(Chunk.id).where(Chunk.doc_id == doc_id))
        return set(rows.scalars().all())


async def test_publish_ingestion_publishes_wrapped_message_to_queue(monkeypatch):
    state = {"declared": 0, "closed": 0, "published": []}

    class _Exchange:
        async def publish(self, message, routing_key):
            state["published"].append((message, routing_key))

    class _Channel:
        def __init__(self):
            self.default_exchange = _Exchange()

    class _Connection:
        async def channel(self):
            return _Channel()

        async def close(self):
            state["closed"] += 1

    async def fake_connect(url):
        assert url == get_settings().rabbitmq_url
        return _Connection()

    async def fake_declare(channel):
        state["declared"] += 1

    monkeypatch.setattr(publisher_module, "connect_robust", fake_connect)
    monkeypatch.setattr(publisher_module, "declare", fake_declare)

    await publish_ingestion(DOC_ID, S3_KEY)

    assert state["declared"] == 1
    assert state["closed"] == 1
    [(message, routing_key)] = state["published"]
    assert routing_key == consumer.QUEUE == "ingestion"
    assert isinstance(message, Message)  # aio-pika 10 rejects raw bytes on publish
    assert json.loads(message.body) == {"doc_id": DOC_ID, "s3_key": S3_KEY}


async def test_publish_ingestion_carries_new_version_when_given(monkeypatch):
    state = {"published": []}

    class _Exchange:
        async def publish(self, message, routing_key):
            state["published"].append((message, routing_key))

    class _Channel:
        def __init__(self):
            self.default_exchange = _Exchange()

    class _Connection:
        async def channel(self):
            return _Channel()

        async def close(self):
            pass

    async def fake_connect(url):
        return _Connection()

    async def fake_declare(channel):
        pass

    monkeypatch.setattr(publisher_module, "connect_robust", fake_connect)
    monkeypatch.setattr(publisher_module, "declare", fake_declare)

    await publish_ingestion(DOC_ID, S3_KEY, new_version=2)

    [(message, routing_key)] = state["published"]
    assert routing_key == consumer.QUEUE
    assert json.loads(message.body) == {"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": 2}


async def test_cancel_document_version_scoped_cleanup_preserves_previous_version(db, monkeypatch):
    chunk_v1 = await _seed_chunk(DOC_ID, version=1)
    await _seed_chunk(DOC_ID, version=2)
    await _seed_doc(
        DOC_ID,
        current_version=1,
        status=DocumentStatus.PROCESSING,
        pending_version=2,
    )

    notices = []
    qdrant_deletes = []

    async def fake_publish(payload, routing_key=consumer.QUEUE):
        notices.append((payload, routing_key))

    def fake_delete_points(point_ids, payload_filter=None):
        qdrant_deletes.append((point_ids, payload_filter))

    monkeypatch.setattr(cancel_module, "publish_message", fake_publish)
    monkeypatch.setattr(cancel_module, "delete_points", fake_delete_points)

    await cancel_module.cancel_document(DOC_ID)

    doc = await _get_doc(DOC_ID)
    assert doc.status == DocumentStatus.FAILED
    assert doc.pending_version is None
    assert await _chunk_ids(DOC_ID) == {chunk_v1}
    assert qdrant_deletes == [
        (
            [],
            {
                "must": [
                    {"key": "doc_id", "match": {"value": DOC_ID}},
                    {"key": "version", "range": {"gte": 2}},
                ]
            },
        )
    ]
    assert notices == [({"doc_id": DOC_ID, "cancel": True}, consumer.QUEUE)]


async def test_cancel_document_first_ingest_removes_partials(db, monkeypatch):
    await _seed_chunk(DOC_ID, version=1)
    await _seed_doc(
        DOC_ID,
        current_version=1,
        status=DocumentStatus.PROCESSING,
        pending_version=1,
    )

    notices = []
    qdrant_deletes = []

    async def fake_publish(payload, routing_key=consumer.QUEUE):
        notices.append((payload, routing_key))

    def fake_delete_points(point_ids, payload_filter=None):
        qdrant_deletes.append((point_ids, payload_filter))

    monkeypatch.setattr(cancel_module, "publish_message", fake_publish)
    monkeypatch.setattr(cancel_module, "delete_points", fake_delete_points)

    await cancel_module.cancel_document(DOC_ID)

    doc = await _get_doc(DOC_ID)
    assert doc.status == DocumentStatus.FAILED
    assert doc.pending_version is None
    assert await _chunk_ids(DOC_ID) == set()
    assert qdrant_deletes == [
        (
            [],
            {
                "must": [
                    {"key": "doc_id", "match": {"value": DOC_ID}},
                    {"key": "version", "range": {"gte": 1}},
                ]
            },
        )
    ]
    assert notices == [({"doc_id": DOC_ID, "cancel": True}, consumer.QUEUE)]


async def test_cancel_document_queued_not_started_cleans_nothing(db, monkeypatch):
    chunk_v1 = await _seed_chunk(DOC_ID, version=1)
    await _seed_doc(DOC_ID, current_version=1, status=DocumentStatus.PENDING)

    notices = []
    qdrant_deletes = []

    async def fake_publish(payload, routing_key=consumer.QUEUE):
        notices.append((payload, routing_key))

    def fake_delete_points(point_ids, payload_filter=None):
        qdrant_deletes.append((point_ids, payload_filter))

    monkeypatch.setattr(cancel_module, "publish_message", fake_publish)
    monkeypatch.setattr(cancel_module, "delete_points", fake_delete_points)

    await cancel_module.cancel_document(DOC_ID)

    doc = await _get_doc(DOC_ID)
    assert doc.status == DocumentStatus.FAILED
    assert doc.pending_version is None
    assert await _chunk_ids(DOC_ID) == {chunk_v1}
    assert qdrant_deletes == []
    assert notices == [({"doc_id": DOC_ID, "cancel": True}, consumer.QUEUE)]


async def test_cancel_document_missing_doc_is_noop(db, monkeypatch):
    async def must_not_publish(payload, routing_key=consumer.QUEUE):
        raise AssertionError("must not publish for unknown doc")

    def must_not_delete_points(point_ids, payload_filter=None):
        raise AssertionError("must not delete for unknown doc")

    monkeypatch.setattr(cancel_module, "publish_message", must_not_publish)
    monkeypatch.setattr(cancel_module, "delete_points", must_not_delete_points)

    await cancel_module.cancel_document("missing-doc-id")


async def test_process_message_cancel_flag_short_circuits(monkeypatch):
    async def must_not_ingest(doc_id, s3_key):
        raise AssertionError("cancelled documents must not be ingested")

    monkeypatch.setattr(consumer, "ingest_document", must_not_ingest)
    await consumer.process_message({"doc_id": DOC_ID, "cancel": True})


def _webhook_client(monkeypatch, published) -> TestClient:
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cr3t")

    async def fake_publish_ingestion(doc_id, s3_key, new_version=None):
        published.append((doc_id, s3_key))

    monkeypatch.setattr(minio_webhook, "publish_ingestion", fake_publish_ingestion)
    app = FastAPI()
    minio_webhook.register_minio_webhook(app)
    register_exception_handlers(app)
    return TestClient(app)


def _webhook_headers(secret: str = "s3cr3t") -> dict:
    return {"X-Webhook-Secret": secret}


def test_minio_event_publishes_ingestion_for_each_record(monkeypatch):
    published = []
    client = _webhook_client(monkeypatch, published)

    resp = client.post(
        "/internal/minio-event",
        json={
            "Records": [
                {"s3": {"object": {"key": "documents/org-1/user-1/doc-1/abc.pdf"}}},
                {"s3": {"object": {"key": ""}}},
                {"not": "an s3 record"},
            ]
        },
        headers=_webhook_headers(),
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert published == [("doc-1", "documents/org-1/user-1/doc-1/abc.pdf")]


def test_minio_event_rejects_missing_secret(monkeypatch):
    published = []
    client = _webhook_client(monkeypatch, published)

    resp = client.post(
        "/internal/minio-event",
        json={"Records": [{"s3": {"object": {"key": "documents/o/u/d/abc.pdf"}}}]},
    )

    assert resp.status_code == 401
    assert "webhook secret" in resp.json()["error"]
    assert published == []


def test_minio_event_rejects_wrong_secret(monkeypatch):
    published = []
    client = _webhook_client(monkeypatch, published)

    resp = client.post(
        "/internal/minio-event",
        json={"Records": [{"s3": {"object": {"key": "documents/o/u/d/abc.pdf"}}}]},
        headers=_webhook_headers("wrong"),
    )

    assert resp.status_code == 401
    assert published == []


def test_minio_event_sets_correlation_id_for_request_logs(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cr3t")
    seen = {}

    async def fake_publish_ingestion(doc_id, s3_key, new_version=None):
        seen["correlation_id"] = get_correlation_id()

    monkeypatch.setattr(minio_webhook, "publish_ingestion", fake_publish_ingestion)
    app = FastAPI()
    minio_webhook.register_minio_webhook(app)
    register_exception_handlers(app)
    client = TestClient(app)

    resp = client.post(
        "/internal/minio-event",
        json={"Records": [{"s3": {"object": {"key": "documents/o/u/d/abc.pdf"}}}]},
        headers={"X-Webhook-Secret": "s3cr3t"},
    )

    assert resp.status_code == 200
    # every log/publish inside the webhook request shares one correlation id
    assert seen["correlation_id"]


def test_minio_event_unconfigured_secret_disables_endpoint(monkeypatch):
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)

    async def must_not_publish(*args, **kwargs):
        raise AssertionError("unauthenticated webhook must not publish")

    monkeypatch.setattr(minio_webhook, "publish_ingestion", must_not_publish)
    app = FastAPI()
    minio_webhook.register_minio_webhook(app)
    register_exception_handlers(app)
    client = TestClient(app)

    # no secret configured: every caller is rejected, even one sending a header
    for headers in (None, {"X-Webhook-Secret": ""}, {"X-Webhook-Secret": "guess"}):
        resp = client.post(
            "/internal/minio-event",
            json={"Records": [{"s3": {"object": {"key": "documents/o/u/d/abc.pdf"}}}]},
            headers=headers,
        )
        assert resp.status_code == 401


def test_minio_event_rejects_unexpected_s3_key(monkeypatch):
    published = []
    client = _webhook_client(monkeypatch, published)

    resp = client.post(
        "/internal/minio-event",
        json={"Records": [{"s3": {"object": {"key": "documents/short.pdf"}}}]},
        headers=_webhook_headers(),
    )

    assert resp.status_code == 500
    assert "unexpected s3 key" in resp.json()["error"]
    assert published == []


def test_minio_event_unquotes_url_encoded_s3_keys(monkeypatch):
    published = []
    client = _webhook_client(monkeypatch, published)

    resp = client.post(
        "/internal/minio-event",
        json={
            "EventName": "s3:ObjectCreated:Put",
            "Key": "documents/documents/org-9/user-9/doc-9/9f8e7d6c.txt",
            "Records": [
                {
                    "eventName": "s3:ObjectCreated:Put",
                    "s3": {
                        "bucket": {"name": "documents"},
                        "object": {
                            "key": "documents%2Forg-9%2Fuser-9%2Fdoc-9%2F9f8e7d6c.txt",
                            "size": 31,
                        },
                    },
                }
            ],
        },
        headers=_webhook_headers(),
    )

    assert resp.status_code == 200
    assert published == [("doc-9", "documents/org-9/user-9/doc-9/9f8e7d6c.txt")]


def test_minio_event_ignores_object_removed_events(monkeypatch):
    published = []
    client = _webhook_client(monkeypatch, published)

    resp = client.post(
        "/internal/minio-event",
        json={
            "Records": [
                {
                    "eventName": "s3:ObjectRemoved:Delete",
                    "s3": {"object": {"key": "documents/org-1/user-1/doc-1/abc.pdf"}},
                },
                {
                    "eventName": "s3:ObjectCreated:Put",
                    "s3": {"object": {"key": "documents/org-1/user-1/doc-2/abc.pdf"}},
                },
            ]
        },
        headers=_webhook_headers(),
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert published == [("doc-2", "documents/org-1/user-1/doc-2/abc.pdf")]


def test_minio_webhook_registered_on_production_app(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "s3cr3t")
    client = TestClient(create_app())
    resp = client.post("/internal/minio-event", json={"Records": []}, headers=_webhook_headers())
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_find_stale_doc_ids_returns_docs_with_stale_chunks(db):
    await _seed_doc(DOC_A, current_version=2)
    await _seed_chunk(DOC_A, version=1)
    await _seed_chunk(DOC_A, version=2)

    await _seed_doc(DOC_B, current_version=2)
    await _seed_chunk(DOC_B, version=1)

    await _seed_doc(DOC_C, current_version=1)
    await _seed_chunk(DOC_C, version=1)

    # DOC_A keeps serving v2 but still has a stale v1 chunk pending cleanup
    assert set(await cleanup_module.find_stale_doc_ids()) == {DOC_A, DOC_B}


async def test_cleanup_job_loop_cleans_stale_docs(monkeypatch):
    cleaned = []

    async def fake_complete_deletions():
        return 0

    async def fake_reap_abandoned_pends():
        return 0

    async def fake_find_stale_doc_ids(limit=100):
        return [DOC_B]

    async def fake_cleanup_stale(doc_id):
        cleaned.append(doc_id)

    async def stop_loop(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(cleanup_module, "complete_deletions", fake_complete_deletions)
    monkeypatch.setattr(cleanup_module, "reap_abandoned_pends", fake_reap_abandoned_pends)
    monkeypatch.setattr(cleanup_module, "find_stale_doc_ids", fake_find_stale_doc_ids)
    monkeypatch.setattr(cleanup_module, "cleanup_stale", fake_cleanup_stale)
    monkeypatch.setattr(cleanup_module.asyncio, "sleep", stop_loop)

    with pytest.raises(asyncio.CancelledError):
        await cleanup_module.cleanup_job_loop(interval_seconds=1)

    assert cleaned == [DOC_B]


async def test_cleanup_job_loop_survives_iteration_failure(monkeypatch):
    attempts = []
    sleeps = []

    async def fake_complete_deletions():
        return 0

    async def fake_reap_abandoned_pends():
        return 0

    async def fake_find_stale_doc_ids(limit=100):
        return [DOC_B, DOC_A]

    async def failing_cleanup_stale(doc_id):
        attempts.append(doc_id)
        raise RuntimeError("qdrant down")

    async def stop_loop(seconds):
        sleeps.append(seconds)
        raise asyncio.CancelledError

    monkeypatch.setattr(cleanup_module, "complete_deletions", fake_complete_deletions)
    monkeypatch.setattr(cleanup_module, "reap_abandoned_pends", fake_reap_abandoned_pends)
    monkeypatch.setattr(cleanup_module, "find_stale_doc_ids", fake_find_stale_doc_ids)
    monkeypatch.setattr(cleanup_module, "cleanup_stale", failing_cleanup_stale)
    monkeypatch.setattr(cleanup_module.asyncio, "sleep", stop_loop)

    with pytest.raises(asyncio.CancelledError):
        await cleanup_module.cleanup_job_loop(interval_seconds=5)

    assert attempts == [DOC_B]
    assert sleeps == [5]
