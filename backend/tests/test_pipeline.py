import hashlib
import io

import pytest
from sqlalchemy import select

import app.db as db_module
from app.core.errors import IngestionError
from app.db import get_session, init_db
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.embed.embeddings import SparseVector
from app.rag.ingestion import pipeline
from app.rag.ingestion.pipeline import (
    cleanup_stale,
    ingest_document,
    ingest_versioned,
)

DOC_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"
ORG_ID = "33333333-3333-3333-3333-333333333333"
S3_KEY = "documents/org/user/doc/sample.txt"

REQUIRED_PAYLOAD_KEYS = {
    "doc_id",
    "user_id",
    "org_id",
    "chunk_text_hash",
    "version",
    "page_number",
    "chunk_index",
    "word_count",
    "char_count",
    "type",
}


@pytest.mark.integration
async def test_ingest_document_returns_chunk_count():
    # requires running MinIO, Postgres, Qdrant and a pre-seeded Document row
    count = await ingest_document("doc-id-1", "documents/org/user/doc-id-1/sample.txt")
    assert count >= 1


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
def fake_s3(monkeypatch):
    objects: dict[str, bytes] = {}

    def fake_get_object(s3_key: str) -> bytes:
        return objects[s3_key]

    monkeypatch.setattr(pipeline, "get_object", fake_get_object)
    return objects


@pytest.fixture
def fake_qdrant(monkeypatch):
    calls = {"upserts": [], "deletes": []}

    def fake_upsert_points_batch(points):
        calls["upserts"].extend(points)

    def fake_delete_points(point_ids, payload_filter=None):
        calls["deletes"].append((point_ids, payload_filter))

    monkeypatch.setattr(pipeline, "upsert_points_batch", fake_upsert_points_batch)
    monkeypatch.setattr(pipeline, "delete_points", fake_delete_points)
    return calls


@pytest.fixture
def fake_embeddings(monkeypatch):
    calls = {"dense": [], "sparse": []}

    async def fake_dense_embed(texts):
        calls["dense"].append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    def fake_sparse_embed(texts):
        calls["sparse"].append(list(texts))
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    monkeypatch.setattr(pipeline, "dense_embed", fake_dense_embed)
    monkeypatch.setattr(pipeline, "sparse_embed", fake_sparse_embed)
    return calls


@pytest.fixture
def fake_cache(monkeypatch):
    patterns: list[str] = []

    class _FakeCache:
        async def delete_by_pattern(self, pattern: str) -> int:
            patterns.append(pattern)
            return 0

    monkeypatch.setattr(pipeline, "get_cache", lambda: _FakeCache())
    return patterns


async def _seed_doc(
    filename: str = "sample.txt",
    content_hash: str = "stale-hash",
    current_version: int = 1,
) -> None:
    async with get_session() as session:
        session.add(
            Document(
                id=DOC_ID,
                user_id=USER_ID,
                org_id=ORG_ID,
                original_filename=filename,
                status=DocumentStatus.PENDING,
                content_hash=content_hash,
                current_version=current_version,
            )
        )
        await session.commit()


async def _seed_chunk(version: int, text: str = "some text") -> str:
    async with get_session() as session:
        chunk = Chunk(
            doc_id=DOC_ID,
            user_id=USER_ID,
            org_id=ORG_ID,
            chunk_text=text,
            page_number=0,
            start_offset=0,
            end_offset=len(text),
            version=version,
        )
        session.add(chunk)
        await session.commit()
        return chunk.id


async def _get_doc() -> Document:
    async with get_session() as session:
        return await session.get(Document, DOC_ID)


async def _chunks_for(doc_id: str = DOC_ID) -> list[Chunk]:
    async with get_session() as session:
        rows = (await session.execute(select(Chunk).where(Chunk.doc_id == doc_id))).scalars().all()
        return list(rows)


def _build_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    for text in ("Alpha beta gamma delta.", "Epsilon zeta eta theta."):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _build_docx() -> bytes:
    from docx import Document as DocxDocument

    buf = io.BytesIO()
    d = DocxDocument()
    d.add_paragraph("Alpha beta gamma delta.")
    d.add_paragraph("Epsilon zeta eta theta.")
    d.save(buf)
    return buf.getvalue()


async def test_ingest_document_persists_chunks_and_payload(
    db, fake_s3, fake_qdrant, fake_embeddings
):
    body = b"Alpha beta gamma delta epsilon zeta eta theta."
    fake_s3[S3_KEY] = body
    await _seed_doc(content_hash="different")

    count = await ingest_document(DOC_ID, S3_KEY)

    assert count >= 1
    doc = await _get_doc()
    assert doc.status == DocumentStatus.EMBEDDED
    assert doc.content_hash == hashlib.sha256(body).hexdigest()

    rows = await _chunks_for()
    assert len(rows) == count
    rows_by_id = {}
    for r in rows:
        assert r.doc_id == DOC_ID
        assert r.user_id == USER_ID
        assert r.org_id == ORG_ID
        assert r.version == 1
        assert r.chunk_text
        rows_by_id[r.id] = r

    assert len(fake_qdrant["upserts"]) == count
    assert {p["id"] for p in fake_qdrant["upserts"]} == set(rows_by_id)
    for point in fake_qdrant["upserts"]:
        payload = point["payload"]
        assert set(payload) >= REQUIRED_PAYLOAD_KEYS
        row = rows_by_id[point["id"]]
        assert payload["doc_id"] == DOC_ID
        assert payload["user_id"] == USER_ID
        assert payload["org_id"] == ORG_ID
        assert payload["version"] == 1
        assert payload["type"] == "txt"
        assert payload["chunk_text_hash"] == hashlib.sha256(row.chunk_text.encode()).hexdigest()
        assert payload["word_count"] == len(row.chunk_text.split())
        assert payload["char_count"] == len(row.chunk_text)
        assert len(point["dense"]) == 2
        assert point["sparse_indices"] == [1]
        assert point["sparse_values"] == [1.0]
    assert fake_embeddings["dense"]


async def test_ingest_document_dedup_returns_zero(db, fake_s3, fake_qdrant, fake_embeddings):
    body = b"same bytes as before"
    fake_s3[S3_KEY] = body
    await _seed_doc(content_hash=hashlib.sha256(body).hexdigest())

    count = await ingest_document(DOC_ID, S3_KEY)

    assert count == 0
    assert (await _get_doc()).status == DocumentStatus.EMBEDDED
    assert fake_qdrant["upserts"] == []
    assert fake_embeddings["dense"] == []
    assert await _chunks_for() == []


async def test_ingest_document_missing_doc_raises(db):
    with pytest.raises(IngestionError):
        await ingest_document("missing-doc-id", S3_KEY)


async def test_ingest_document_wraps_parser_errors(db, fake_s3, fake_qdrant, fake_embeddings):
    fake_s3[S3_KEY] = b""
    await _seed_doc(filename="broken.pdf", content_hash="different")

    with pytest.raises(IngestionError) as excinfo:
        await ingest_document(DOC_ID, S3_KEY)

    assert "parse failed" in excinfo.value.detail
    assert excinfo.value.__cause__ is not None
    assert not isinstance(excinfo.value.__cause__, IngestionError)
    assert (await _get_doc()).status == DocumentStatus.FAILED


async def test_ingest_document_skips_marker_only_chunks(db, fake_s3, fake_qdrant, fake_embeddings):
    fake_s3[S3_KEY] = b"# Intro\n\nBody text goes here.\n\n\n\n"
    await _seed_doc(content_hash="different")

    count = await ingest_document(DOC_ID, S3_KEY)

    rows = await _chunks_for()
    assert count == len(rows) == 2
    assert {r.chunk_text for r in rows} == {"Intro", "Body text goes here."}
    assert len(fake_qdrant["upserts"]) == 2


async def test_ingest_document_all_chunks_empty_returns_zero(
    db, fake_s3, fake_qdrant, fake_embeddings
):
    fake_s3[S3_KEY] = b"   \n\n \t \n"
    await _seed_doc(content_hash="different")

    count = await ingest_document(DOC_ID, S3_KEY)

    assert count == 0
    assert await _chunks_for() == []
    assert fake_qdrant["upserts"] == []
    assert fake_embeddings["dense"] == []
    assert (await _get_doc()).status == DocumentStatus.EMBEDDED


async def test_ingest_document_marks_failed_on_embedding_error(
    db, fake_s3, fake_qdrant, monkeypatch
):
    fake_s3[S3_KEY] = b"Alpha beta gamma delta."
    await _seed_doc(content_hash="different")

    async def failing_dense_embed(texts):
        raise RuntimeError("embed down")

    monkeypatch.setattr(pipeline, "dense_embed", failing_dense_embed)
    with pytest.raises(RuntimeError):
        await ingest_document(DOC_ID, S3_KEY)

    assert (await _get_doc()).status == DocumentStatus.FAILED
    assert await _chunks_for() == []


async def test_ingest_failure_deletes_partial_qdrant_points(
    db, fake_s3, fake_qdrant, fake_embeddings, monkeypatch
):
    body = "\n\n".join(f"Paragraph {i} with a few words." for i in range(40)).encode()
    fake_s3[S3_KEY] = body
    await _seed_doc(content_hash="different")

    upsert_calls = []

    def failing_upsert_points_batch(points):
        upsert_calls.append(len(points))
        if len(upsert_calls) >= 2:
            raise RuntimeError("qdrant down mid-batch")

    monkeypatch.setattr(pipeline, "upsert_points_batch", failing_upsert_points_batch)

    with pytest.raises(RuntimeError):
        await ingest_document(DOC_ID, S3_KEY)

    assert upsert_calls == [32, 8]
    assert (await _get_doc()).status == DocumentStatus.FAILED
    assert await _chunks_for() == []
    assert len(fake_qdrant["deletes"]) == 1
    point_ids, payload_filter = fake_qdrant["deletes"][0]
    assert point_ids == []
    assert payload_filter == {
        "must": [
            {"key": "doc_id", "match": {"value": DOC_ID}},
            {"key": "version", "match": {"value": 1}},
        ]
    }


async def test_ingest_pdf_roundtrip_per_page(db, fake_s3, fake_qdrant, fake_embeddings):
    fake_s3[S3_KEY] = _build_pdf()
    await _seed_doc(filename="doc.pdf", content_hash="different")

    count = await ingest_document(DOC_ID, S3_KEY)

    rows = await _chunks_for()
    assert count == len(rows) == 2
    assert sorted(r.page_number for r in rows) == [0, 1]
    texts_by_page = {r.page_number: r.chunk_text for r in rows}
    assert texts_by_page[0] == "Alpha beta gamma delta."
    assert texts_by_page[1] == "Epsilon zeta eta theta."
    for point in fake_qdrant["upserts"]:
        assert point["payload"]["type"] == "pdf"
        assert point["payload"]["page_number"] in (0, 1)


async def test_ingest_docx_roundtrip_single_page(db, fake_s3, fake_qdrant, fake_embeddings):
    fake_s3[S3_KEY] = _build_docx()
    await _seed_doc(filename="doc.docx", content_hash="different")

    count = await ingest_document(DOC_ID, S3_KEY)

    rows = await _chunks_for()
    assert count == len(rows) == 1
    assert rows[0].page_number == 0
    assert "Alpha beta gamma delta." in rows[0].chunk_text
    assert fake_qdrant["upserts"][0]["payload"]["type"] == "docx"


async def test_ingest_versioned_writes_new_version_then_flips(
    db, fake_s3, fake_qdrant, fake_embeddings, fake_cache
):
    fake_s3[S3_KEY] = b"Version two content with fresh words."
    await _seed_doc(current_version=1)

    count = await ingest_versioned(DOC_ID, S3_KEY, 2)

    assert count >= 1
    rows = await _chunks_for()
    assert len(rows) == count
    assert {r.version for r in rows} == {2}
    doc = await _get_doc()
    assert doc.current_version == 2
    assert doc.status == DocumentStatus.EMBEDDED
    assert fake_cache == [f"retrieval:{DOC_ID}:*"]
    for point in fake_qdrant["upserts"]:
        assert point["payload"]["version"] == 2


async def test_ingest_versioned_reingests_unchanged_content_at_new_version(
    db, fake_s3, fake_qdrant, fake_embeddings, fake_cache
):
    body = b"Identical content across versions."
    fake_s3[S3_KEY] = body
    await _seed_doc(content_hash=hashlib.sha256(body).hexdigest(), current_version=1)

    count = await ingest_versioned(DOC_ID, S3_KEY, 2)

    assert count >= 1
    assert {r.version for r in await _chunks_for()} == {2}
    assert (await _get_doc()).current_version == 2
    assert fake_cache == [f"retrieval:{DOC_ID}:*"]


async def test_ingest_versioned_failure_keeps_current_version(
    db, fake_s3, fake_qdrant, fake_cache, monkeypatch
):
    fake_s3[S3_KEY] = b"Version two content."
    await _seed_doc(current_version=1)

    async def failing_dense_embed(texts):
        raise RuntimeError("embed down")

    monkeypatch.setattr(pipeline, "dense_embed", failing_dense_embed)
    with pytest.raises(RuntimeError):
        await ingest_versioned(DOC_ID, S3_KEY, 2)

    doc = await _get_doc()
    assert doc.current_version == 1
    assert doc.status == DocumentStatus.FAILED
    assert await _chunks_for() == []
    assert fake_cache == []


async def test_cleanup_stale_deletes_old_versions(db, fake_qdrant):
    await _seed_doc(current_version=2)
    stale_id = await _seed_chunk(version=1, text="old text")
    kept_id = await _seed_chunk(version=2, text="new text")

    removed = await cleanup_stale(DOC_ID)

    assert removed == 1
    assert {r.id for r in await _chunks_for()} == {kept_id}
    assert fake_qdrant["deletes"] == [
        ([stale_id], {"must": [{"key": "doc_id", "match": {"value": DOC_ID}}]})
    ]


async def test_cleanup_stale_missing_doc_returns_zero(db, fake_qdrant):
    assert await cleanup_stale("missing-doc-id") == 0
    assert fake_qdrant["deletes"] == []
