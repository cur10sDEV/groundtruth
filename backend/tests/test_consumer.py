import pytest

from app.core.errors import IngestionError
from app.ingestion import consumer
from app.ingestion.consumer import process_message
from app.models.document import Document, DocumentStatus


def test_process_message_signature():
    # unit-safety: no queue required for signature; integration behavior needs RabbitMQ
    assert callable(process_message)


DOC_ID = "11111111-1111-1111-1111-111111111111"
ORG_ID = "33333333-3333-3333-3333-333333333333"
USER_ID = "22222222-2222-2222-2222-222222222222"
S3_KEY = f"documents/{ORG_ID}/{USER_ID}/{DOC_ID}/sample.txt"


def _doc(status: DocumentStatus, current_version: int = 1) -> Document:
    return Document(
        id=DOC_ID,
        user_id=USER_ID,
        org_id=ORG_ID,
        original_filename="sample.txt",
        status=status,
        content_hash="hash",
        current_version=current_version,
    )


def _patch_session(monkeypatch, doc):
    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def get(self, model, pk):
            return doc

    monkeypatch.setattr(consumer, "get_session", _Session)


async def test_process_message_missing_fields_raises():
    with pytest.raises(IngestionError) as excinfo:
        await process_message({})
    assert "missing doc_id" in excinfo.value.detail

    with pytest.raises(IngestionError) as excinfo:
        await process_message({"doc_id": DOC_ID})
    assert "missing s3_key" in excinfo.value.detail


async def test_process_message_unknown_doc_raises(monkeypatch):
    _patch_session(monkeypatch, None)
    with pytest.raises(IngestionError) as excinfo:
        await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY})
    assert "unknown doc" in excinfo.value.detail


async def test_process_message_embedded_short_circuits(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.EMBEDDED))
    calls = []

    async def must_not_ingest(doc_id, s3_key):
        calls.append((doc_id, s3_key))

    monkeypatch.setattr(consumer, "ingest_document", must_not_ingest)
    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY})
    assert calls == []


async def test_process_message_failed_doc_skips_ingestion(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.FAILED))
    calls = []

    async def must_not_ingest(doc_id, s3_key):
        calls.append((doc_id, s3_key))

    monkeypatch.setattr(consumer, "ingest_document", must_not_ingest)
    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY})
    assert calls == []


async def test_process_message_deleting_doc_returns_without_ingesting(monkeypatch):
    # deletion is linearized: a DELETING doc must never ingest, plain or versioned
    _patch_session(monkeypatch, _doc(DocumentStatus.DELETING))
    calls = []

    async def must_not_ingest(doc_id, s3_key):
        calls.append(("plain", doc_id))

    async def must_not_ingest_versioned(doc_id, s3_key, new_version):
        calls.append(("versioned", doc_id))

    monkeypatch.setattr(consumer, "ingest_document", must_not_ingest)
    monkeypatch.setattr(consumer, "ingest_versioned", must_not_ingest_versioned)

    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY})
    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": 2})

    assert calls == []


async def test_process_message_calls_ingest_document(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.PENDING))
    calls = []

    async def fake_ingest(doc_id, s3_key):
        calls.append((doc_id, s3_key))
        return 3

    monkeypatch.setattr(consumer, "ingest_document", fake_ingest)
    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY})
    assert calls == [(DOC_ID, S3_KEY)]


async def test_process_message_s3_key_owner_mismatch_raises(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.PENDING))
    calls = []

    async def must_not_ingest(doc_id, s3_key):
        calls.append(doc_id)
        raise AssertionError("cross-tenant key must never be ingested")

    monkeypatch.setattr(consumer, "ingest_document", must_not_ingest)

    bad_key = f"documents/other-org/{USER_ID}/{DOC_ID}/sample.txt"
    with pytest.raises(IngestionError) as excinfo:
        await process_message({"doc_id": DOC_ID, "s3_key": bad_key})
    assert "does not match document owner" in excinfo.value.detail

    bad_key = f"documents/{ORG_ID}/other-user/{DOC_ID}/sample.txt"
    with pytest.raises(IngestionError):
        await process_message({"doc_id": DOC_ID, "s3_key": bad_key})
    assert calls == []


async def test_process_message_malformed_s3_key_raises(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.PENDING))
    with pytest.raises(IngestionError) as excinfo:
        await process_message({"doc_id": DOC_ID, "s3_key": "not-a-valid-key"})
    assert "malformed s3 key" in excinfo.value.detail


async def test_process_message_versioned_routes_to_ingest_versioned(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.PENDING, current_version=1))
    calls = []

    async def fake_ingest_versioned(doc_id, s3_key, new_version):
        calls.append((doc_id, s3_key, new_version))
        return 2

    async def must_not_ingest(doc_id, s3_key):
        raise AssertionError("plain path must not run for versioned messages")

    monkeypatch.setattr(consumer, "ingest_versioned", fake_ingest_versioned)
    monkeypatch.setattr(consumer, "ingest_document", must_not_ingest)

    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": 2})

    assert calls == [(DOC_ID, S3_KEY, 2)]


async def test_process_message_versioned_failed_doc_proceeds(monkeypatch):
    # a versioned message is also the recovery path for a FAILED attempt
    _patch_session(monkeypatch, _doc(DocumentStatus.FAILED, current_version=1))
    calls = []

    async def fake_ingest_versioned(doc_id, s3_key, new_version):
        calls.append(doc_id)
        return 2

    monkeypatch.setattr(consumer, "ingest_versioned", fake_ingest_versioned)

    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": 2})

    assert calls == [DOC_ID]


async def test_process_message_versioned_skips_when_flip_already_applied(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.EMBEDDED, current_version=2))
    calls = []

    async def must_not_ingest_versioned(doc_id, s3_key, new_version):
        calls.append(doc_id)
        raise AssertionError("must not re-apply an already-applied version")

    monkeypatch.setattr(consumer, "ingest_versioned", must_not_ingest_versioned)

    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": 2})

    assert calls == []


async def test_process_message_versioned_invalid_version_raises():
    with pytest.raises(IngestionError) as excinfo:
        await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": "two"})
    assert "invalid new_version" in excinfo.value.detail

    with pytest.raises(IngestionError):
        await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY, "new_version": 0})
