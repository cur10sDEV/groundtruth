# Presigned Ingestion Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement these plans task-by-task. Tasks use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace multipart upload + webhook trigger with presigned-POST direct-to-MinIO uploads, a MinIO→RabbitMQ event trigger with a validating translator, true document deletion, user-scoped duplicate detection, and reaper sweeps.

**Architecture:** Browsers PUT nothing through the API — a signing endpoint returns a presigned POST (size-capped by MinIO policy); MinIO notifies RabbitMQ directly (AMQP, disk-backed); a translator task in the worker validates events and republishes clean messages to the `ingestion` queue. Deletion linearizes with one atomic transaction (status=DELETING + chunk delete) and finishes idempotently (reaper completes tombstones); all pipeline status writes become guarded updates so a delete/cancel can never be outraced by an in-flight ingest.

**Tech Stack:** Existing stack only — boto3 presigned POST, aio-pika, MinIO `notify_amqp`, SQLAlchemy/Alembic, Next.js 14.

**Spec:** `docs/superpowers/specs/2026-09-25-presigned-ingestion-design.md` (the plan argues from the spec; executors read both)

## Global Constraints

- Every step ships green: from `backend/` run `./.venv/bin/python -m pytest tests/ -v` (baseline 281 passed) + `./.venv/bin/python -m ruff check . && ./.venv/bin/python -m ruff format --check .`; frontend gates `npx tsc --noEmit` + `npm run build` from `frontend/`.
- Unit tests are offline (monkeypatch at module boundaries per existing conventions); live-stack tests are marked `integration`.
- Migrations must be idempotent both ways (fresh DB via 0001's create_all, upgraded DB via guarded DDL) — follow the 0002 guarded pattern; sqlite (test fixture) must no-op Postgres-only DDL.
- `DELETING` may never be stomped by any other status write (guarded updates only).
- The duplicate check is `(org_id, user_id)`-scoped, `status == EMBEDDED`, `id != self`.
- MinIO event config is injected via compose env (`RABBIT_USER`/`RABBIT_PASS`, defaults guest/guest); no new compose services.
- No plan-doc history rewrites: living docs only (README).

---

### Task 1: Foundations — DELETING status, migration 0003, config, events metric

**Files:**
- Modify: `backend/app/models/document.py`
- Create: `backend/alembic/versions/0003_deleting_status.py`
- Modify: `backend/app/core/config.py`, `backend/.env.example`
- Modify: `backend/app/core/metrics.py`
- Test: `backend/tests/test_foundations.py`

**Interfaces:**
- Produces: `DocumentStatus.DELETING`; `Document.failure_reason: Mapped[str | None]`; config `upload_max_bytes: int = 52428800`, `presign_expiry_seconds: int = 900`, `reaper_pending_after_seconds: int = 3600`; metric `EVENTS_DROPPED` (Counter, label `reason`).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_foundations.py`:
```python
from app.core.config import get_settings
from app.core.metrics import EVENTS_DROPPED
from app.models.document import DocumentStatus


def test_deleting_status_exists():
    assert DocumentStatus.DELETING.value == "DELETING"


def test_new_settings_defaults():
    s = get_settings()
    assert s.upload_max_bytes == 50 * 1024 * 1024
    assert s.presign_expiry_seconds == 900
    assert s.reaper_pending_after_seconds == 3600


def test_events_dropped_counter_has_reason_label():
    from prometheus_client import generate_latest

    EVENTS_DROPPED.labels(reason="test").inc()
    assert b"rag_events_dropped_total" in generate_latest()
    assert b'reason="test"' in generate_latest()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_foundations.py -v`
Expected: FAIL (`DELETING` / fields / metric missing).

- [ ] **Step 3: Implement**

`backend/app/models/document.py` — add to the enum and the model:
```python
class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    EMBEDDED = "EMBEDDED"
    FAILED = "FAILED"
    DELETING = "DELETING"
```
and after `pending_version`:
```python
    failure_reason: Mapped[str | None] = mapped_column(String(1024))
```

`backend/app/core/config.py` — add (and mirror in `backend/.env.example` with `UPLOAD_MAX_BYTES=52428800`, `PRESIGN_EXPIRY_SECONDS=900`, `REAPER_PENDING_AFTER_SECONDS=3600`):
```python
    upload_max_bytes: int = 50 * 1024 * 1024
    presign_expiry_seconds: int = 900
    reaper_pending_after_seconds: int = 3600
```

`backend/app/core/metrics.py`:
```python
EVENTS_DROPPED = Counter("rag_events_dropped_total", "Ingestion events dropped by the translator", ["reason"])
```

`backend/alembic/versions/0003_deleting_status.py` — follow 0002's guarded style:
```python
"""add DELETING status + documents.failure_reason

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None

_TABLE = "documents"
_COLUMN = "failure_reason"


def _column_exists(insp, table: str, column: str) -> bool:
    return any(c["name"] == column for c in insp.get_columns(table))


def _enum_value_exists(conn, value: str) -> bool:
    if conn.dialect.name != "postgresql":
        return True  # sqlite fixture: create_all already has everything
    return bool(
        conn.execute(
            sa.text(
                "SELECT 1 FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
                "WHERE t.typname = 'documentstatus' AND e.enumlabel = :v"
            ),
            {"v": value},
        ).first()
    )


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if bind.dialect.name == "postgresql" and not _enum_value_exists(bind, "DELETING"):
        op.execute("ALTER TYPE documentstatus ADD VALUE 'DELETING'")
    if not _column_exists(insp, _TABLE, _COLUMN):
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(1024), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if _column_exists(insp, _TABLE, _COLUMN):
        op.drop_column(_TABLE, _COLUMN)
    # enum value removal is not supported in postgres; leave DELETING in place
```

- [ ] **Step 4: Run tests + migrations to verify**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_foundations.py tests/ -v` — all green.
Run: `./.venv/bin/python -m alembic upgrade head && ./.venv/bin/python -m alembic downgrade base && ./.venv/bin/python -m alembic upgrade head` (sqlite: passes; Postgres idempotency is live-verified in Task 8).

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/document.py backend/alembic/versions/0003_deleting_status.py \
        backend/app/core/config.py backend/.env.example backend/app/core/metrics.py backend/tests/test_foundations.py
git commit -m "feat(ingestion): add DELETING status, failure_reason, presign/reaper config, events metric"
```

---

### Task 2: S3 helpers — presign_upload + delete_prefix

**Files:**
- Modify: `backend/app/core/s3.py`
- Test: `backend/tests/test_s3.py` (extend)

**Interfaces:**
- Consumes: Task 1's `upload_max_bytes`, `presign_expiry_seconds`.
- Produces: `presign_upload(org_id: str, user_id: str, doc_id: str, ext: str) -> dict` returning `{"url", "fields", "key"}`; `delete_prefix(org_id: str, user_id: str, doc_id: str) -> int` (count of objects deleted).

- [ ] **Step 1: Write the failing tests** (extend `backend/tests/test_s3.py`; stub `_client` at module boundary)

```python
from app.core import s3 as s3mod


class _FakePresignClient:
    def generate_presigned_post(self, Bucket, Key, Conditions, ExpiresIn):
        self.calls = {"bucket": Bucket, "key": Key, "conditions": Conditions, "expires": ExpiresIn}
        return {"url": "http://minio/documents", "fields": {"key": Key, "x": "1"}}


class _FakeListClient:
    def __init__(self, keys):
        self.keys = keys
        self.deleted = []

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.keys if k.startswith(Prefix)]}

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.deleted.append(obj["Key"])
        return {"Deleted": Delete["Objects"]}


def test_presign_upload_shape(monkeypatch):
    fake = _FakePresignClient()
    monkeypatch.setattr(s3mod, "_client", lambda: fake)
    out = s3mod.presign_upload("o", "u", "d", "pdf")
    assert out["url"] == "http://minio/documents"
    assert out["fields"]["key"].startswith("documents/o/u/d/")
    assert out["fields"]["key"].endswith(".pdf")
    assert fake.calls["conditions"] == [["content-length-range", 0, get_settings().upload_max_bytes]]
    assert fake.calls["expires"] == get_settings().presign_expiry_seconds


def test_delete_prefix_deletes_all_versions(monkeypatch):
    fake = _FakeListClient(["documents/o/u/d/a.pdf", "documents/o/u/d/b.pdf", "documents/o/u/e/c.pdf"])
    monkeypatch.setattr(s3mod, "_client", lambda: fake)
    assert s3mod.delete_prefix("o", "u", "d") == 2
    assert sorted(fake.deleted) == ["documents/o/u/d/a.pdf", "documents/o/u/d/b.pdf"]


def test_delete_prefix_empty_is_noop(monkeypatch):
    fake = _FakeListClient([])
    monkeypatch.setattr(s3mod, "_client", lambda: fake)
    assert s3mod.delete_prefix("o", "u", "d") == 0
    fake.delete_objects  # never called: no batches
```
Note: import `get_settings` in the test file as existing tests do.

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ./.venv/bin/python -m pytest tests/test_s3.py -v` — new tests FAIL (functions missing).

- [ ] **Step 3: Implement** — add to `backend/app/core/s3.py`:

```python
from uuid import uuid4  # already imported

def presign_upload(org_id: str, user_id: str, doc_id: str, ext: str) -> dict:
    s = get_settings()
    client = _client()
    key = build_key(org_id, user_id, doc_id, f"{uuid4()}.{ext}")
    post = client.generate_presigned_post(
        Bucket=s.s3_bucket,
        Key=key,
        Conditions=[["content-length-range", 0, s.upload_max_bytes]],
        ExpiresIn=s.presign_expiry_seconds,
    )
    return {"url": post["url"], "fields": post["fields"], "key": key}


def delete_prefix(org_id: str, user_id: str, doc_id: str) -> int:
    s = get_settings()
    client = _client()
    prefix = f"documents/{org_id}/{user_id}/{doc_id}/"
    deleted = 0
    keys: list[str] = []
    resp = client.list_objects_v2(Bucket=s.s3_bucket, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", [])]
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        client.delete_objects(Bucket=s.s3_bucket, Delete={"Objects": [{"Key": k} for k in batch]})
        deleted += len(batch)
    return deleted
```

- [ ] **Step 4: Run tests** — `cd backend && ./.venv/bin/python -m pytest tests/test_s3.py tests/ -v` all green; ruff gate.

- [ ] **Step 5: Commit** — `git commit -m "feat(s3): presigned POST uploads and prefix deletion helpers"`

---

### Task 3: Pipeline — guarded status transitions (the race keystone)

**Files:**
- Modify: `backend/app/rag/ingestion/pipeline.py`
- Test: `backend/tests/test_pipeline.py` (extend)

**Interfaces:**
- Produces (internal, used by Tasks 4–6): `_mark_failed(doc_id, reason=None)` — guarded `UPDATE documents SET status='FAILED', pending_version=NULL, failure_reason=:r WHERE id=:d AND status='PROCESSING'`; `_finalize(doc_id, content_hash, new_version=None)` — guarded `UPDATE ... SET status='EMBEDDED', pending_version=NULL, content_hash=:h[, current_version=:nv] WHERE id=:d AND status='PROCESSING'`, raises `IngestionError` on rowcount 0. Public API unchanged.

**Restructure (exact behavior):**
1. `_ingest` start: replace the direct `doc.status = PROCESSING` write with a guarded claim: `UPDATE documents SET status='PROCESSING', pending_version=:v WHERE id=:d AND status IN ('PENDING','PROCESSING','FAILED')` — rowcount 0 → raise `IngestionError("document not processing-claimable")`. FAILED must remain claimable because the consumer's versioned-recovery path deliberately routes versioned messages to FAILED docs (the plain path never reaches `_ingest` for FAILED rows — the consumer guards it); DELETING (and EMBEDDED) must never be claimable.
2. Dedup short-circuit (same-doc, same-hash): instead of mutating the ORM object, call `_finalize(doc_id, content_hash=doc.content_hash)`; rowcount 0 → raise (lost the race) → except path handles.
3. Success path: the chunk-rows commit keeps ONLY chunks (drop the `doc.status = EMBEDDED / pending_version = None / content_hash` mutations from it); after it, call `_finalize(doc_id, content_hash, new_version=version)`. This makes the guarded `_finalize` THE atomic flip for both paths (plain: no version change; versioned: `current_version=new_version`), matching spec §9.
4. Both failure handlers: replace the `doc.status = FAILED` mutations with `_mark_failed(doc_id)`; keep `_cleanup_partial_qdrant` calls; `_mark_failed` must not raise on rowcount 0 (lost race is fine — DELETING won) and must not increment `INGESTION_FAILED` on lost-race (rowcount 0) paths; keep `INGESTION_FAILED.inc()` in the handler as today.
5. Guarded updates via SQLAlchemy core on a fresh session: `from sqlalchemy import update`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_delete_during_ingest_cannot_be_flipped(db, seeded_doc):
    # simulate: a DELETING row while _ingest runs
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        d.status = DocumentStatus.DELETING
        await s.commit()
    # claim must fail: DELETING row cannot be re-claimed
    with pytest.raises(IngestionError):
        await _ingest(seeded_doc.id, "documents/o/u/d/x.pdf")


async def test_finalize_loses_race_to_deleting(db, seeded_doc, monkeypatch):
    # doc PROCESSING, chunks committed, then deletion flips status mid-run
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        d.status = DocumentStatus.PROCESSING
        d.pending_version = 1
        await s.commit()
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        d.status = DocumentStatus.DELETING
        await s.commit()
    with pytest.raises(IngestionError):
        await _finalize(seeded_doc.id, "hash123")
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        assert d.status == DocumentStatus.DELETING  # not stomped


async def test_mark_failed_does_not_stomp_deleting(db, seeded_doc):
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        d.status = DocumentStatus.DELETING
        await s.commit()
    await _mark_failed(seeded_doc.id, reason="boom")
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        assert d.status == DocumentStatus.DELETING
        assert d.failure_reason is None
```
(Use the existing test_pipeline.py sqlite fixtures/helpers; adapt names to what exists there.)

- [ ] **Step 2: Run to verify failure** — `./.venv/bin/python -m pytest tests/test_pipeline.py -v` — new tests FAIL; existing suite must be fixed by Step 3 (the restructure intentionally moves finalization).

- [ ] **Step 3: Implement the restructure** per the Interfaces block above. Also update the flip in `ingest_versioned` — it disappears: `_ingest` finalizes internally; `ingest_versioned` becomes `_ingest(...) + _invalidate_cache(...)` (keep the public signature).

- [ ] **Step 4: Run the FULL suite** — `./.venv/bin/python -m pytest tests/ -v` all green (281 baseline may shift by the moved expectations; nothing may be deleted, only adapted honestly); ruff gate.

- [ ] **Step 5: Commit** — `git commit -m "feat(ingestion): guarded status transitions make deletion un-raceable"`

---

### Task 4: Pipeline — cross-document duplicate detection (user-scoped)

**Files:**
- Modify: `backend/app/rag/ingestion/pipeline.py`
- Test: `backend/tests/test_pipeline.py` (extend)

**Interfaces:**
- Consumes: Task 3's `_mark_failed`; Task 1's `failure_reason`.
- Produces: duplicate behavior in `_ingest` (after same-doc dedup, before `_reset_version`).

- [ ] **Step 1: Write the failing tests**

```python
async def test_cross_doc_duplicate_marks_failed_with_reason(db, seeded_doc, other_embedded_doc, monkeypatch):
    # other_embedded_doc: same (org, user), same content_hash, different id, EMBEDDED
    deleted = []
    monkeypatch.setattr("app.rag.ingestion.pipeline.get_object", lambda k: b"same-bytes")
    monkeypatch.setattr("app.rag.ingestion.pipeline.delete_object", lambda k: deleted.append(k))
    monkeypatch.setattr("app.rag.ingestion.pipeline._reset_version", AsyncMock())
    n = await _ingest(seeded_doc.id, seeded_doc.s3_key)
    assert n == 0
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        assert d.status == DocumentStatus.FAILED
        assert d.failure_reason == f"duplicate of {other_embedded_doc.id}"
    assert deleted == [seeded_doc.s3_key]
    _reset_version not called  # assert via the mock


async def test_duplicate_check_is_user_scoped(db, seeded_doc, other_user_same_hash_doc, monkeypatch):
    # other_user_same_hash_doc: same org, DIFFERENT user, same content_hash, EMBEDDED → NOT a duplicate
    async with get_session() as s:
        other = Document(
            id="other-user-doc", user_id="user-b", org_id=seeded_doc.org_id,
            original_filename="other.pdf", status=DocumentStatus.EMBEDDED,
            content_hash=<the seeded content hash>, current_version=1,
        )
        s.add(other)
        await s.commit()
    # stub fetch to return the same bytes the OTHER doc hashed; stub embed as existing tests do
    n = await _ingest(seeded_doc.id, seeded_doc.s3_key)
    assert n > 0  # ingest proceeded — no duplicate flag
    async with get_session() as s:
        d = await s.get(Document, seeded_doc.id)
        assert d.status == DocumentStatus.EMBEDDED
        assert d.failure_reason is None
```
Plus two companion tests with the same harness: `test_duplicate_check_excludes_self` (call `_ingest(doc.id, key, version=doc.current_version + 1)` with bytes identical to the doc's own current content — must NOT be flagged; it re-ingests as the new version) and `test_duplicate_ignores_non_embedded` (seed a PENDING doc with the same hash, not EMBEDDED — must NOT be flagged). Use existing pipeline-test stubbing helpers for `get_object`, `dense_embed`, `sparse_embed`, `parse_bytes`, and Qdrant fakes; where the existing tests seed documents, reuse those helpers (read `tests/test_pipeline.py` first).

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement** — in `_ingest`, after the same-doc dedup block and before `await _reset_version(...)`:

```python
            dup = (
                (
                    await session.execute(
                        select(Document.id).where(
                            Document.content_hash == content_hash,
                            Document.org_id == doc.org_id,
                            Document.user_id == doc.user_id,
                            Document.id != doc_id,
                            Document.status == DocumentStatus.EMBEDDED,
                        )
                    )
                )
                .scalars()
                .first()
            )
            if dup is not None:
                try:
                    delete_object(s3_key)
                except Exception:
                    logger.warning("duplicate blob cleanup failed", extra={"s3_key": s3_key})
                await _mark_failed(doc_id, reason=f"duplicate of {dup}")
                return 0
```
Import `delete_object` from `app.core.s3` alongside `get_object`. Note: session is still open; `_mark_failed` uses its own fresh session (as built in Task 3).

- [ ] **Step 4: Run full suite + ruff; Step 5: Commit** — `git commit -m "feat(ingestion): user-scoped cross-document duplicate detection"`

---

### Task 5: True deletion — route cascade, list filter, cancel guard

**Files:**
- Modify: `backend/app/api/routes_documents.py`, `backend/app/ingestion/cancel.py`
- Test: `backend/tests/test_documents.py` (extend)

**Interfaces:**
- Consumes: Task 1 `DELETING`; Task 2 `delete_prefix`; existing `delete_points`, `invalidate_for_doc`, `_mark_failed`-style guarded writes.
- Produces: `DELETE /documents/{id}` cascade; `GET /documents` excludes `DELETING`; reaper consumes the tombstone (Task 6).

- [ ] **Step 1: Write the failing tests**

```python
async def test_delete_cascade_removes_everything(client, embedded_doc, fake_qdrant, fake_cache, fake_s3):
    r = await client.delete(f"/documents/{embedded_doc.id}", headers=auth)
    assert r.status_code == 200 and r.json()["status"] == "deleted"
    # row gone, chunks gone, qdrant delete_points called with doc_id filter,
    # cache invalidated, s3 delete_prefix called


async def test_delete_partial_failure_leaves_tombstone(client, embedded_doc, monkeypatch):
    monkeypatch.setattr(routes_documents, "delete_prefix", AsyncMock(side_effect=StorageError(detail="boom")))
    r = await client.delete(...)
    assert r.status_code == 200 and r.json()["status"] == "deleting"
    # chunks gone (Tx1), row still present with status DELETING


async def test_list_excludes_deleting(client, ...): ...
async def test_cancel_never_resurrects_deleting(client, ...): ...
```
(Adapt to the existing test_documents.py harness: fixtures `client`, seeding helpers, `fake_*` stubs per its conventions.)

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

In `routes_documents.py` replace the delete route:
```python
@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(get_current_user)) -> dict:
    from app.core.qdrant_store import delete_points
    from app.core.s3 import delete_prefix

    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
        await session.execute(
            update(Document)
            .where(Document.id == doc_id, Document.org_id == user["org_id"])
            .values(status=DocumentStatus.DELETING, failure_reason=None)
        )
        await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        await session.commit()
    done = True
    try:
        delete_points([], {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})
    except Exception:
        logger.warning("delete: qdrant cleanup failed", extra={"doc_id": doc_id})
        done = False
    try:
        await invalidate_for_doc(user["org_id"], doc_id)
    except Exception:
        logger.warning("delete: cache invalidation failed", extra={"doc_id": doc_id})
        done = False
    try:
        delete_prefix(user["org_id"], user["user_id"], doc_id)
    except Exception:
        logger.warning("delete: blob cleanup failed", extra={"doc_id": doc_id})
        done = False
    if done:
        async with get_session() as session:
            d = await session.get(Document, doc_id)
            if d is not None:
                await session.delete(d)
                await session.commit()
    return {"doc_id": doc_id, "status": "deleted" if done else "deleting"}
```
In `list_documents` add `Document.status != DocumentStatus.DELETING` to the where clause. In `cancel.py`, guard the status write: replace `doc.status = DocumentStatus.FAILED` with a core update `WHERE Document.status.in_((PENDING, PROCESSING))` (rowcount 0 → log + return; never touch DELETING rows; keep the pending-version point cleanup as-is).

- [ ] **Step 4: Run full suite + ruff; Step 5: Commit** — `git commit -m "feat(api): true document deletion with linearizing tombstone"`

---

### Task 6: Reaper extensions — abandoned PENDING + DELETING completion

**Files:**
- Modify: `backend/app/ingestion/cleanup_job.py`
- Test: `backend/tests/test_publisher.py` (extend, where the cleanup tests live) or a new `backend/tests/test_cleanup_job.py`

**Interfaces:**
- Consumes: Tasks 1/2/5 (`DELETING`, `delete_prefix`, config `reaper_pending_after_seconds`).
- Produces: `reap_abandoned_pends() -> int`, `complete_deletions() -> int` — called each `cleanup_job_loop` iteration alongside the stale sweep.

- [ ] **Step 1: Write the failing tests**

```python
async def test_reap_abandoned_pending_deletes_row_and_blobs(db, old_pending_doc, monkeypatch):
    monkeypatch.setattr(cleanup_job, "delete_prefix", lambda *a: 1)
    assert await cleanup_job.reap_abandoned_pends() == 1
    # row gone; a fresh PENDING doc is untouched (created_at under threshold)


async def test_complete_deletions_finishes_tombstone(db, deleting_doc, monkeypatch):
    # deleting_doc: status DELETING, stale points/cache/blob present
    # monkeypatch delete_points / invalidate_for_doc / delete_prefix at module boundary
    assert await cleanup_job.complete_deletions() == 1
    # row gone; idempotent: second call returns 0
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement**

In `cleanup_job.py`:
```python
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.core.config import get_settings
from app.core.qdrant_store import delete_points
from app.core.s3 import delete_prefix
from app.db import get_session
from app.models.document import Document, DocumentStatus
from app.rag.retrieval.cache import invalidate_for_doc


async def reap_abandoned_pends() -> int:
    cutoff = datetime.now(UTC) - timedelta(seconds=get_settings().reaper_pending_after_seconds)
    async with get_session() as session:
        rows = (
            (
                await session.execute(
                    select(Document).where(
                        Document.status == DocumentStatus.PENDING,
                        Document.created_at < cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
        for d in rows:
            try:
                delete_prefix(d.org_id, d.user_id, d.id)
            except Exception:
                logger.warning("reaper blob cleanup failed", extra={"doc_id": d.id})
            await session.delete(d)
        await session.commit()
        return len(rows)


async def complete_deletions() -> int:
    async with get_session() as session:
        rows = (
            (
                await session.execute(select(Document).where(Document.status == DocumentStatus.DELETING))
            )
            .scalars()
            .all()
        )
        for d in rows:
            try:
                delete_points([], {"must": [{"key": "doc_id", "match": {"value": d.id}}]})
                await invalidate_for_doc(d.org_id, d.id)
                delete_prefix(d.org_id, d.user_id, d.id)
            except Exception:
                logger.warning("deletion completion failed; will retry", extra={"doc_id": d.id})
                continue
            await session.delete(d)
        await session.commit()
        return len(rows)
```
In `cleanup_job_loop`, each iteration: `await complete_deletions(); await reap_abandoned_pends();` then the existing stale sweep.

- [ ] **Step 4: Run full suite + ruff; Step 5: Commit** — `git commit -m "feat(ingestion): reaper for abandoned uploads and deletion tombstones"`

---

### Task 7: Signing endpoint (additive — multipart stays until Task 8)

**Files:**
- Modify: `backend/app/api/routes_documents.py`
- Test: `backend/tests/test_documents.py` (extend)

**Interfaces:**
- Consumes: Task 2 `presign_upload`; existing versioning logic.
- Produces: `POST /documents/sign` body `{"filename": str}` → `200 {"doc_id", "status", "new_version"?, "upload": {"url", "fields"}}`. Sets the same PENDING/pending_version state as the old upload route. Does NOT publish anything.

- [ ] **Step 1: Write the failing tests**

```python
async def test_sign_new_document(client, auth, monkeypatch):
    monkeypatch.setattr(routes_documents, "presign_upload", lambda *a, **k: {"url": "u", "fields": {"key": "k"}, "key": "documents/o/u/d/x.pdf"})
    r = await client.post("/documents/sign", json={"filename": "policy.pdf"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["upload"]["url"] == "u"
    # row created PENDING, current_version 1, pending_version None


async def test_sign_versioned_reupload(client, embedded_doc, auth, monkeypatch): ...
    # same filename EMBEDDED → same doc_id, status pending, new_version == current+1,
    # pending_version set, no second row


async def test_sign_requires_auth(client): ...  # 401 without token
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement**

```python
class SignIn(BaseModel):
    filename: str


@router.post("/sign")
async def sign_upload(body: SignIn, user: dict = Depends(get_current_user)) -> dict:
    filename = body.filename or "untitled"
    if "/" in filename or "\\" in filename:
        raise ValidationError(detail="invalid filename")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    async with get_session() as session:
        versioned = (
            (
                await session.execute(
                    select(Document)
                    .where(
                        Document.org_id == user["org_id"],
                        Document.user_id == user["user_id"],
                        Document.original_filename == filename,
                        Document.status == DocumentStatus.EMBEDDED,
                    )
                    .order_by(Document.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        if versioned is not None:
            new_version = versioned.current_version + 1
            versioned.status = DocumentStatus.PENDING
            versioned.pending_version = new_version
            await session.commit()
            presigned = presign_upload(user["org_id"], user["user_id"], versioned.id, ext)
            return {
                "doc_id": versioned.id,
                "status": "pending",
                "new_version": new_version,
                "upload": {"url": presigned["url"], "fields": presigned["fields"]},
            }
        doc = Document(
            user_id=user["user_id"],
            org_id=user["org_id"],
            original_filename=filename,
            status=DocumentStatus.PENDING,
            content_hash="",  # unknown until the worker fetches the object
            current_version=1,
        )
        session.add(doc)
        await session.commit()
        presigned = presign_upload(user["org_id"], user["user_id"], doc.id, ext)
        return {
            "doc_id": doc.id,
            "status": "pending",
            "upload": {"url": presigned["url"], "fields": presigned["fields"]},
        }
```
Import `presign_upload` from `app.core.s3` and `BaseModel` from `pydantic` at the top of the file (module-level import so tests can monkeypatch `routes_documents.presign_upload`).

- [ ] **Step 4: Run full suite + ruff; Step 5: Commit** — `git commit -m "feat(api): presigned upload signing endpoint"`

---

### Task 8: Trigger handover — translator, worker wiring, consumer guard, MinIO init, multipart removal

**Files:**
- Create: `backend/app/ingestion/event_translator.py`
- Modify: `backend/app/ingestion/worker.py`, `backend/app/ingestion/consumer.py`, `backend/app/api/routes_documents.py` (remove `/upload` multipart route), `infra/minio-init/init-bucket.sh`, `infra/docker-compose.yml`
- Test: `backend/tests/test_event_translator.py`; extend `backend/tests/test_consumer.py`

**Interfaces:**
- Consumes: `publish_ingestion` (existing), `key_to_parts` (existing), Task 1 `EVENTS_DROPPED`, Task 5's `DELETING`.
- Produces: `translate_loop() -> None` (coroutine); consumer skips `DELETING`; `POST /documents/upload` is REMOVED (frontend switches in Task 9 — 8 and 9 land adjacently).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_event_translator.py`:
```python
def _event(key="documents/o/u/d/x.pdf", name="s3:ObjectCreated:Post", bucket="documents"):
    return {
        "EventName": name,
        "s3": {"bucket": {"name": bucket}, "object": {"key": key}},
    }


async def test_valid_event_forwards_to_ingestion(db, pending_doc, monkeypatch):
    published = []
    async def fake_publish(doc_id, s3_key, new_version=None):
        published.append((doc_id, s3_key, new_version))
    monkeypatch.setattr(event_translator, "publish_ingestion", fake_publish)
    out = await event_translator.translate_event(_event(key=pending_doc.s3_key))
    assert out is True
    assert published == [(pending_doc.id, pending_doc.s3_key, None)]


async def test_drop_reasons(db, pending_doc, caplog, monkeypatch):
    # not ObjectCreated / wrong bucket / malformed key / missing row /
    # non-PENDING row (EMBEDDED + DELETING) / ownership mismatch → False
    # and EVENTS_DROPPED.labels(reason=...) incremented (assert via counter read)
```
Extend `test_consumer.py`: `DELETING` doc → `process_message` returns without calling ingest (both plain and versioned messages).

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

`backend/app/ingestion/event_translator.py`:
```python
import asyncio
import json
import logging

from aio_pika import connect_robust

from app.core.metrics import EVENTS_DROPPED
from app.core.s3 import key_to_parts
from app.db import get_session
from app.ingestion.publisher import publish_ingestion
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)
EVENTS_QUEUE = "minio.events"
EVENTS_EXCHANGE = "minio.events"
FORWARDABLE = {DocumentStatus.PENDING, DocumentStatus.PROCESSING}


def _drop(reason: str, key: str = "") -> bool:
    EVENTS_DROPPED.labels(reason=reason).inc()
    logger.warning("event dropped", extra={"reason": reason, "key": key})
    return False


async def translate_event(event: dict) -> bool:
    name = str(event.get("EventName", ""))
    if not name.startswith("s3:ObjectCreated:"):
        return _drop("not_created", name)
    s3 = event.get("s3") or {}
    if (s3.get("bucket") or {}).get("name") != get_settings().s3_bucket:
        return _drop("wrong_bucket")
    key = str((s3.get("object") or {}).get("key", ""))
    try:
        key_org, key_user, key_doc, _ = key_to_parts(key)
    except ValueError:
        return _drop("malformed_key", key)
    async with get_session() as session:
        doc = await session.get(Document, key_doc)
        if doc is None:
            return _drop("missing_row", key)
        if doc.status not in FORWARDABLE:
            return _drop("not_pending", key)
        if doc.org_id != key_org or doc.user_id != key_user:
            return _drop("ownership_mismatch", key)
        new_version = doc.pending_version
    await publish_ingestion(key_doc, key, new_version=new_version)
    return True


async def translate_loop() -> None:
    settings = get_settings()
    connection = await connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    exchange = await channel.declare_exchange(EVENTS_EXCHANGE, ExchangeType.DIRECT, durable=True)
    queue = await channel.declare_queue(EVENTS_QUEUE, durable=True)
    await queue.bind(exchange, routing_key=EVENTS_QUEUE)
    async with queue.iterator() as qiter:
        async for message in qiter:
            try:
                await translate_event(json.loads(message.body))
            except Exception:
                logger.exception("translate failed", extra={"body": message.body})
            finally:
                await message.ack()  # noise never retries — drops are terminal
```
Imports: `get_settings` from `app.core.config`, `ExchangeType` from `aio_pika`. Wire in `worker.py`:
```python
from app.ingestion.event_translator import translate_loop
await asyncio.gather(consume_loop(), translate_loop(), cleanup_job_loop())
```
Consumer guard (in `process_message`, before the versioned/plain branches):
```python
        if doc.status == DocumentStatus.DELETING:
            return
```
`infra/minio-init/init-bucket.sh` — append (idempotent; `mc admin config set` is a no-op when unchanged; guard the restart+event add with a marker check `mc event ls local/documents arn:minio:sqs::primary:amqp | grep -q put ||` before `mc event add`):
```sh
mc admin config set local notify_amqp:primary \
  url="amqp://${RABBIT_USER:-guest}:${RABBIT_PASS:-guest}@rabbitmq:5672" \
  exchange="minio.events" exchange_type="direct" routing_key="minio.events" \
  durable="on" queue_dir="/tmp/minio-events" queue_limit="10000"
mc admin service restart local
mc event add local/documents arn:minio:sqs::primary:amqp --event put --prefix documents/ --ignore-existing
```
`infra/docker-compose.yml`: `minio-init` gains `depends_on: rabbitmq: {condition: service_healthy}` and environment `RABBIT_USER: guest`, `RABBIT_PASS: guest`. Remove the `/upload` multipart route (and its `put_object`/`publish_ingestion` imports) from `routes_documents.py`.

- [ ] **Step 4: Live verification (record outputs)** — boot `docker compose -f infra/docker-compose.yml up -d rabbitmq minio minio-init`, wait for init exit 0, verify `docker compose exec minio mc event ls local/documents` shows the put trigger (or `mc` alias inside the init container); then from `backend/`: `.venv/bin/python` script that presigns (Task 2 helper), uploads bytes via `requests.post(url, data=fields, files={"file": ...})`, and asserts the translated `{doc_id, s3_key}` message lands on the `ingestion` queue (`rabbitmqctl list_queues` or aio-pika consume). Mark this as an integration test file `backend/tests/test_integration_events.py` with `@pytest.mark.integration` if feasible; otherwise record manual steps + outputs in the report. Stop+rm the started services afterward.

- [ ] **Step 5: Run full suite + ruff; Commit** — `git commit -m "feat(ingestion): minio->rabbitmq trigger with validating translator; remove multipart upload"`

---

### Task 9: Frontend switch — sign + direct POST, failure_reason

**Files:**
- Modify: `frontend/components/UploadButton.tsx`, `frontend/components/DocumentList.tsx`
- Test: `npx tsc --noEmit` + `npm run build` (component behavior is thin; no FE unit framework is configured)

**Interfaces:**
- Consumes: Task 7's `POST /documents/sign` response `{"doc_id", "status", "new_version"?, "upload": {"url", "fields"}}`; documents list now includes `failure_reason` for FAILED docs.

- [ ] **Step 1: Backend list shape** — in `routes_documents.py` `list_documents` add `"failure_reason": d.failure_reason` to the returned dicts (one line; the GET-by-id route too).

- [ ] **Step 2: UploadButton switch**

```tsx
const sign = async (file: File) => {
  const signRes = await fetch(`${API_URL}/documents/sign`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ filename: file.name }),
  });
  if (!signRes.ok) throw new Error("sign failed");
  const { upload } = await signRes.json();
  const form = new FormData();
  Object.entries(upload.fields as Record<string, string>).forEach(([k, v]) => form.append(k, v));
  form.append("file", file);
  const up = await fetch(upload.url, { method: "POST", body: form });
  if (!up.ok) throw new Error(`upload failed: ${up.status}`);
  onUploaded?.();
};
```
Keep the existing client-side size guard if present (mirror 50MB).

- [ ] **Step 3: DocumentList** — skip rows with `status === "DELETING"`; for `FAILED` rows render `failure_reason` (link to the referenced doc when the string matches `duplicate of {id}`). Use the existing row-render style.

- [ ] **Step 4: Verify** — `cd frontend && npx tsc --noEmit && npm run build` green. Backend full suite + ruff green (list-shape change may need test updates — update honestly).

- [ ] **Step 5: Commit** — `git commit -m "feat(frontend): presigned direct uploads and duplicate reasons"`

---

### Task 10: Webhook removal + README + final E2E

**Files:**
- Delete: `backend/app/ingestion/minio_webhook.py` + its tests (locate via `rg -l minio_webhook backend/tests`)
- Modify: `backend/app/main.py` (remove registration), `backend/app/core/config.py` + `backend/.env.example` (remove `WEBHOOK_SECRET`), `README.md`

**Steps:**
- [ ] **Step 1:** Remove the webhook module, its route registration in `main.py`, `webhook_secret` config + `WEBHOOK_SECRET` env example, and all its tests. Grep-verify zero references: `rg -i "minio_webhook|webhook_secret|WEBHOOK_SECRET" backend/` → empty.
- [ ] **Step 2:** README updates: upload flow (sign → direct POST), trigger (MinIO→RabbitMQ events, translator, `rag_events_dropped_total`), deletion semantics (DELETING + reaper), duplicate behavior (FAILED + reason), remove all webhook mentions; keep the quickstart commands verified.
- [ ] **Step 3:** Full gates: backend suite + ruff; frontend tsc + build; `docker compose -f infra/docker-compose.yml config -q`.
- [ ] **Step 4: Live E2E (record outputs):** boot rabbitmq+minio+minio-init; start the worker (`python -m app.ingestion.worker`); `uvicorn` the API; signup via curl; sign; upload; watch the doc go EMBEDDED; re-upload same filename → version 2; upload duplicate bytes under a new name → FAILED + `duplicate of {id}`; DELETE → verify chunks/points/blobs gone (`mc ls` prefix empty). Stop started services.
- [ ] **Step 5: Commit** — `git commit -m "chore: remove minio webhook; document event-driven ingestion"`

---

**Exit check:** full suite green; tsc/build green; merged compose config clean; live E2E recorded: sign→upload→event→ingest→EMBEDDED, versioned re-upload, duplicate FAILED, delete cascade; `rg -i webhook backend/ infra/ README.md` empty.
