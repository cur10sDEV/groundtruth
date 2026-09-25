import logging

import pytest
from prometheus_client import generate_latest

import app.db as db_module
from app.db import get_session, init_db
from app.ingestion import event_translator
from app.models.document import Document, DocumentStatus

DOC_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"
ORG_ID = "33333333-3333-3333-3333-333333333333"
S3_KEY = f"documents/{ORG_ID}/{USER_ID}/{DOC_ID}/sample.txt"
MISSING_DOC = "00000000-0000-0000-0000-000000000000"


def _event(key="documents/o/u/d/x.pdf", name="s3:ObjectCreated:Post", bucket="documents"):
    return {
        "EventName": name,
        "s3": {"bucket": {"name": bucket}, "object": {"key": key}},
    }


def _dropped(reason: str) -> float:
    prefix = f'rag_events_dropped_total{{reason="{reason}"}} '
    for line in generate_latest().decode().splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


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


@pytest.fixture
async def pending_doc(db):
    async with get_session() as session:
        doc = Document(
            id=DOC_ID,
            user_id=USER_ID,
            org_id=ORG_ID,
            original_filename="sample.txt",
            status=DocumentStatus.PENDING,
            content_hash="hash",
            current_version=1,
        )
        session.add(doc)
        await session.commit()
    doc.s3_key = S3_KEY
    return doc


@pytest.fixture
def no_publish(monkeypatch):
    published = []

    async def fake_publish(doc_id, s3_key, new_version=None):
        published.append((doc_id, s3_key, new_version))

    monkeypatch.setattr(event_translator, "publish_ingestion", fake_publish)
    return published


async def test_valid_event_forwards_to_ingestion(db, pending_doc, monkeypatch):
    published = []

    async def fake_publish(doc_id, s3_key, new_version=None):
        published.append((doc_id, s3_key, new_version))

    monkeypatch.setattr(event_translator, "publish_ingestion", fake_publish)
    out = await event_translator.translate_event(_event(key=pending_doc.s3_key))
    assert out is True
    assert published == [(pending_doc.id, pending_doc.s3_key, None)]


async def test_processing_doc_forwards_pending_version(db, no_publish):
    async with get_session() as session:
        doc = Document(
            id=DOC_ID,
            user_id=USER_ID,
            org_id=ORG_ID,
            original_filename="sample.txt",
            status=DocumentStatus.PROCESSING,
            content_hash="hash",
            current_version=1,
            pending_version=2,
        )
        session.add(doc)
        await session.commit()
        doc.s3_key = S3_KEY

    assert await event_translator.translate_event(_event(key=doc.s3_key)) is True
    assert no_publish == [(DOC_ID, S3_KEY, 2)]


async def test_minio_record_shape_and_encoded_key_forwards(db, pending_doc, no_publish):
    # real MinIO AMQP payloads: the S3 record lives under Records[0] and the
    # object key arrives percent-encoded
    event = {
        "EventName": "s3:ObjectCreated:Post",
        "Key": pending_doc.s3_key,
        "Records": [
            {
                "eventName": "s3:ObjectCreated:Post",
                "s3": {
                    "bucket": {"name": "documents"},
                    "object": {"key": pending_doc.s3_key.replace("/", "%2F")},
                },
            }
        ],
    }
    assert await event_translator.translate_event(event) is True
    assert no_publish == [(DOC_ID, S3_KEY, None)]


async def test_drop_reasons(db, pending_doc, no_publish, caplog):
    cases = [
        ("not_created", _event(name="s3:ObjectRemoved:Delete"), "s3:ObjectRemoved:Delete"),
        ("not_created", _event(name=""), ""),
        ("wrong_bucket", _event(bucket="other-bucket"), ""),
        ("malformed_key", _event(key="documents/short.pdf"), "documents/short.pdf"),
        (
            "missing_row",
            _event(key=f"documents/{ORG_ID}/{USER_ID}/{MISSING_DOC}/x.pdf"),
            f"documents/{ORG_ID}/{USER_ID}/{MISSING_DOC}/x.pdf",
        ),
        (
            "ownership_mismatch",
            _event(key=f"documents/other-org/{USER_ID}/{DOC_ID}/x.pdf"),
            f"documents/other-org/{USER_ID}/{DOC_ID}/x.pdf",
        ),
        (
            "ownership_mismatch",
            _event(key=f"documents/{ORG_ID}/other-user/{DOC_ID}/x.pdf"),
            f"documents/{ORG_ID}/other-user/{DOC_ID}/x.pdf",
        ),
    ]
    with caplog.at_level(logging.WARNING):
        for reason, event, _ in cases:
            before = _dropped(reason)
            assert await event_translator.translate_event(event) is False
            assert _dropped(reason) == before + 1
    assert no_publish == []
    drops = [r for r in caplog.records if "event dropped" in r.message]
    assert [(r.reason, r.key) for r in drops] == [
        (reason, expected_key) for reason, _, expected_key in cases
    ]


async def test_drop_not_pending_for_embedded_and_deleting(db, no_publish):
    key = None
    for doc_id, status in (
        ("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee", DocumentStatus.EMBEDDED),
        ("dddddddd-dddd-dddd-dddd-dddddddddddd", DocumentStatus.DELETING),
    ):
        key = f"documents/{ORG_ID}/{USER_ID}/{doc_id}/x.pdf"
        async with get_session() as session:
            session.add(
                Document(
                    id=doc_id,
                    user_id=USER_ID,
                    org_id=ORG_ID,
                    original_filename="sample.txt",
                    status=status,
                    content_hash="hash",
                    current_version=1,
                )
            )
            await session.commit()

        before = _dropped("not_pending")
        assert await event_translator.translate_event(_event(key=key)) is False
        assert _dropped("not_pending") == before + 1
    assert no_publish == []
