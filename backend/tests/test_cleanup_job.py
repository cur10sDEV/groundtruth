import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

import app.db as db_module
from app.core.config import get_settings
from app.db import get_session, init_db
from app.ingestion import cleanup_job as cleanup_module
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus

DOC_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"
ORG_ID = "33333333-3333-3333-3333-333333333333"
DOC_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


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
    status: DocumentStatus = DocumentStatus.PENDING,
    created_at: datetime | None = None,
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
                current_version=1,
                created_at=created_at,
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


async def _get_doc(doc_id: str) -> Document | None:
    async with get_session() as session:
        return await session.get(Document, doc_id)


async def _chunk_ids(doc_id: str) -> set[str]:
    async with get_session() as session:
        rows = await session.execute(select(Chunk.id).where(Chunk.doc_id == doc_id))
        return set(rows.scalars().all())


def _old_created_at() -> datetime:
    # old enough to cross the reaper threshold regardless of the configured value
    return datetime.now(UTC) - timedelta(seconds=get_settings().reaper_pending_after_seconds * 2)


async def test_reap_abandoned_pends_deletes_old_pending_row_and_blobs(db, monkeypatch):
    # sqlite stores naive-UTC datetimes (tzinfo dropped on bind) and its
    # CURRENT_TIMESTAMP is UTC too, so the aware-UTC cutoff compares correctly.
    await _seed_doc(DOC_ID, created_at=_old_created_at())
    await _seed_doc(DOC_B)

    prefix_deletes = []

    def fake_delete_prefix(org_id, user_id, doc_id):
        prefix_deletes.append((org_id, user_id, doc_id))
        return 1

    monkeypatch.setattr(cleanup_module, "delete_prefix", fake_delete_prefix)

    assert await cleanup_module.reap_abandoned_pends() == 1

    assert await _get_doc(DOC_ID) is None
    assert await _get_doc(DOC_B) is not None
    assert prefix_deletes == [(ORG_ID, USER_ID, DOC_ID)]


async def test_reap_abandoned_pends_blob_failure_still_deletes_row(db, monkeypatch):
    await _seed_doc(DOC_ID, created_at=_old_created_at())

    def failing_delete_prefix(org_id, user_id, doc_id):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(cleanup_module, "delete_prefix", failing_delete_prefix)

    assert await cleanup_module.reap_abandoned_pends() == 1
    assert await _get_doc(DOC_ID) is None


async def test_reap_abandoned_pends_ignores_fresh_pending_and_other_statuses(db, monkeypatch):
    await _seed_doc(DOC_ID, status=DocumentStatus.EMBEDDED, created_at=_old_created_at())
    await _seed_doc(DOC_B, status=DocumentStatus.PENDING)

    monkeypatch.setattr(cleanup_module, "delete_prefix", lambda *a: 0)

    assert await cleanup_module.reap_abandoned_pends() == 0
    assert await _get_doc(DOC_ID) is not None
    assert await _get_doc(DOC_B) is not None


def _mock_externals(monkeypatch, failing: str | None = None) -> list[str]:
    calls: list[str] = []

    def fake_delete_points(point_ids, payload_filter=None):
        calls.append("points")
        if failing == "delete_points":
            raise RuntimeError("qdrant down")

    async def fake_invalidate_for_doc(org_id, doc_id):
        calls.append("cache")
        if failing == "invalidate_for_doc":
            raise RuntimeError("redis down")

    def fake_delete_prefix(org_id, user_id, doc_id):
        calls.append("prefix")
        if failing == "delete_prefix":
            raise RuntimeError("s3 down")

    monkeypatch.setattr(cleanup_module, "delete_points", fake_delete_points)
    monkeypatch.setattr(cleanup_module, "invalidate_for_doc", fake_invalidate_for_doc)
    monkeypatch.setattr(cleanup_module, "delete_prefix", fake_delete_prefix)
    return calls


async def test_complete_deletions_finishes_tombstone(db, monkeypatch):
    await _seed_doc(DOC_ID, status=DocumentStatus.DELETING)
    await _seed_chunk(DOC_ID, version=1)

    calls = _mock_externals(monkeypatch)

    assert await cleanup_module.complete_deletions() == 1
    assert await cleanup_module.complete_deletions() == 0  # idempotent

    assert await _get_doc(DOC_ID) is None
    assert await _chunk_ids(DOC_ID) == set()
    assert calls == ["points", "cache", "prefix"]

    args = {}

    def spy_points(point_ids, payload_filter=None):
        args["points"] = (point_ids, payload_filter)

    async def spy_cache(org_id, doc_id):
        args["cache"] = (org_id, doc_id)

    def spy_prefix(org_id, user_id, doc_id):
        args["prefix"] = (org_id, user_id, doc_id)

    monkeypatch.setattr(cleanup_module, "delete_points", spy_points)
    monkeypatch.setattr(cleanup_module, "invalidate_for_doc", spy_cache)
    monkeypatch.setattr(cleanup_module, "delete_prefix", spy_prefix)

    await _seed_doc(DOC_B, status=DocumentStatus.DELETING)
    assert await cleanup_module.complete_deletions() == 1
    assert args["points"] == ([], {"must": [{"key": "doc_id", "match": {"value": DOC_B}}]})
    assert args["cache"] == (ORG_ID, DOC_B)
    assert args["prefix"] == (ORG_ID, USER_ID, DOC_B)


async def test_complete_deletions_sweeps_orphan_chunks_before_externals(db, monkeypatch):
    await _seed_doc(DOC_ID, status=DocumentStatus.DELETING)
    await _seed_chunk(DOC_ID, version=1)

    order: list[str] = []
    real_delete = cleanup_module.delete

    def recording_delete(*args, **kwargs):
        order.append("chunks")
        return real_delete(*args, **kwargs)

    def fake_delete_points(point_ids, payload_filter=None):
        order.append("points")

    async def fake_invalidate_for_doc(org_id, doc_id):
        order.append("cache")

    def fake_delete_prefix(org_id, user_id, doc_id):
        order.append("prefix")

    monkeypatch.setattr(cleanup_module, "delete", recording_delete)
    monkeypatch.setattr(cleanup_module, "delete_points", fake_delete_points)
    monkeypatch.setattr(cleanup_module, "invalidate_for_doc", fake_invalidate_for_doc)
    monkeypatch.setattr(cleanup_module, "delete_prefix", fake_delete_prefix)

    assert await cleanup_module.complete_deletions() == 1

    assert order == ["chunks", "points", "cache", "prefix"]
    assert await _chunk_ids(DOC_ID) == set()


@pytest.mark.parametrize("failing", ["delete_points", "invalidate_for_doc", "delete_prefix"])
async def test_complete_deletions_step_failure_leaves_tombstone_then_retry_completes(
    db, monkeypatch, failing
):
    await _seed_doc(DOC_ID, status=DocumentStatus.DELETING)
    await _seed_chunk(DOC_ID, version=1)

    calls = _mock_externals(monkeypatch, failing=failing)

    assert await cleanup_module.complete_deletions() == 0

    doc = await _get_doc(DOC_ID)
    assert doc is not None
    assert doc.status == DocumentStatus.DELETING
    step_of = {"delete_points": 0, "invalidate_for_doc": 1, "delete_prefix": 2}
    assert calls == ["points", "cache", "prefix"][: step_of[failing] + 1]

    _mock_externals(monkeypatch)
    assert await cleanup_module.complete_deletions() == 1
    assert await _get_doc(DOC_ID) is None


async def test_cleanup_job_loop_runs_reaper_then_stale_sweep(monkeypatch):
    calls = []

    async def fake_complete_deletions():
        calls.append("complete_deletions")

    async def fake_reap():
        calls.append("reap")

    async def fake_find_stale_doc_ids(limit=100):
        calls.append("find_stale")
        return [DOC_B]

    async def fake_cleanup_stale(doc_id):
        calls.append(("cleanup", doc_id))

    async def stop_loop(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(cleanup_module, "complete_deletions", fake_complete_deletions)
    monkeypatch.setattr(cleanup_module, "reap_abandoned_pends", fake_reap)
    monkeypatch.setattr(cleanup_module, "find_stale_doc_ids", fake_find_stale_doc_ids)
    monkeypatch.setattr(cleanup_module, "cleanup_stale", fake_cleanup_stale)
    monkeypatch.setattr(cleanup_module.asyncio, "sleep", stop_loop)

    with pytest.raises(asyncio.CancelledError):
        await cleanup_module.cleanup_job_loop(interval_seconds=1)

    assert calls == ["complete_deletions", "reap", "find_stale", ("cleanup", DOC_B)]
