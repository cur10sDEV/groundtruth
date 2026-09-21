import pytest

from app.core.errors import IngestionError
from app.ingestion import consumer
from app.ingestion.consumer import process_message
from app.models.document import Document, DocumentStatus


def test_process_message_signature():
    # unit-safety: no queue required for signature; integration behavior needs RabbitMQ
    assert callable(process_message)


DOC_ID = "11111111-1111-1111-1111-111111111111"
S3_KEY = "documents/org/user/doc/sample.txt"


def _doc(status: DocumentStatus) -> Document:
    return Document(
        id=DOC_ID,
        user_id="22222222-2222-2222-2222-222222222222",
        org_id="33333333-3333-3333-3333-333333333333",
        original_filename="sample.txt",
        status=status,
        content_hash="hash",
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
    assert "missing doc_id/s3_key" in excinfo.value.detail

    with pytest.raises(IngestionError) as excinfo:
        await process_message({"doc_id": DOC_ID})
    assert "missing doc_id/s3_key" in excinfo.value.detail


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


async def test_process_message_calls_ingest_document(monkeypatch):
    _patch_session(monkeypatch, _doc(DocumentStatus.PENDING))
    calls = []

    async def fake_ingest(doc_id, s3_key):
        calls.append((doc_id, s3_key))
        return 3

    monkeypatch.setattr(consumer, "ingest_document", fake_ingest)
    await process_message({"doc_id": DOC_ID, "s3_key": S3_KEY})
    assert calls == [(DOC_ID, S3_KEY)]
