# Phase 1 — Storage Layer

**Goal:** Implement the data layer: SQLAlchemy models + migrations (users, orgs, memberships,
documents, chunks, citations, query_logs, cache_index), the MinIO (S3) client, the Qdrant client +
collection setup, and Redis clients for semantic cache and rate limiting.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md` (sections 4, 8)

## Dependencies

Phase 0 (config, logging, errors, docker-compose). Backing services must be running
(`docker compose -f infra/docker-compose.yml up -d`).

---

### Task 1.1: SQLAlchemy models + Alembic migrations

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/base.py`
- Create: `backend/app/models/user.py`
- Create: `backend/app/models/organization.py`
- Create: `backend/app/models/document.py`
- Create: `backend/app/models/chunk.py`
- Create: `backend/app/models/citation.py`
- Create: `backend/app/models/query_log.py`
- Create: `backend/app/models/cache_index.py`
- Create: `backend/app/db.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/versions/0001_initial.py`
- Create: `backend/alembic/versions/0002_add_pending_version.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Consumes: `get_settings()`.
- Produces:
  - `Base` (DeclarativeBase) in `backend/app/models/base.py`.
  - Async engine + session factory `get_session()` (async context manager yielding a
    `AsyncSession`) in `backend/app/db.py`; `init_db()` to create tables for tests.
  - Models (SQLAlchemy 2.0 mapped):
    - `User(id, email, password_hash, created_at)`
    - `Organization(id, name, created_at)`
    - `Membership(id, user_id, org_id, role)`
    - `Document(id, user_id, org_id, original_filename, status, content_hash, current_version,
      pending_version, created_at, updated_at)` — `pending_version: int | None` is the
      version of the in-flight ingest attempt (set by the pipeline alongside `PROCESSING`,
      cleared on every exit); it scopes cancellation cleanup to the partial attempt.
    - `Chunk(id, doc_id, user_id, org_id, chunk_text, page_number, start_offset, end_offset,
      version, created_at, updated_at)`
    - `Citation(id, query_id, chunk_id, doc_id)`
    - `QueryLog(id, query_id, user_id, org_id, query, cached, trace_id, created_at)`
    - `CacheIndex(id, doc_id, cache_key)`
  - Enum `DocumentStatus` in `backend/app/models/document.py` with members `PENDING`, `PROCESSING`,
    `EMBEDDED`, `FAILED`.
  - Enum `Role` in `backend/app/models/organization.py` with `OWNER`, `ADMIN`, `MEMBER`.

- [ ] **Step 1: Write the failing model test**

`backend/tests/test_models.py`:
```python
import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import init_db, get_session
from app.models.document import Document, DocumentStatus


@pytest.fixture
async def session():
    await init_db()
    async with get_session() as s:
        yield s


async def test_document_round_trip(session: AsyncSession):
    doc = Document(
        id="11111111-1111-1111-1111-111111111111",
        user_id="22222222-2222-2222-2222-222222222222",
        org_id="33333333-3333-3333-3333-333333333333",
        original_filename="a.txt",
        status=DocumentStatus.PENDING,
        content_hash="abc",
        current_version=1,
    )
    session.add(doc)
    await session.commit()

    rows = (await session.execute(
        __import__("sqlalchemy").select(Document)
    )).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == DocumentStatus.PENDING
    assert rows[0].current_version == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: app.models`.

- [ ] **Step 3: Write models, db session, and migration**

`backend/app/models/base.py`:
```python
from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UUIDPkMixin:
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
```

`backend/app/models/user.py`:
```python
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class User(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
```

`backend/app/models/organization.py`:
```python
from enum import StrEnum

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class Role(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class Organization(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "organizations"
    name: Mapped[str] = mapped_column(String(255))


class Membership(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "memberships"
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), index=True
    )
    org_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organizations.id"), index=True
    )
    role: Mapped[Role] = mapped_column(Enum(Role))
```

`backend/app/models/document.py`:
```python
from enum import StrEnum

from sqlalchemy import Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    EMBEDDED = "EMBEDDED"
    FAILED = "FAILED"


class Document(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    org_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organizations.id"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(1024))
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    pending_version: Mapped[int | None] = mapped_column(Integer)
```

`backend/app/models/chunk.py`:
```python
from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class Chunk(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "chunks"
    doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int] = mapped_column(Integer, default=0)
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
```

`backend/app/models/citation.py`:
```python
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPkMixin


class Citation(UUIDPkMixin, Base):
    __tablename__ = "citations"
    query_id: Mapped[str] = mapped_column(String(36), index=True)
    chunk_id: Mapped[str] = mapped_column(String(36), ForeignKey("chunks.id"), index=True)
    doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
```

`backend/app/models/query_log.py`:
```python
from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPkMixin


class QueryLog(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "query_logs"
    query_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"))
    query: Mapped[str] = mapped_column(Text)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    trace_id: Mapped[str] = mapped_column(String(36))
```

`backend/app/models/cache_index.py`:
```python
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPkMixin


class CacheIndex(UUIDPkMixin, Base):
    __tablename__ = "cache_index"
    doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), index=True)
    cache_key: Mapped[str] = mapped_column(String(128), index=True)
```

`backend/app/models/__init__.py`:
```python
from app.models.base import Base
from app.models.cache_index import CacheIndex
from app.models.chunk import Chunk
from app.models.citation import Citation
from app.models.document import Document
from app.models.organization import Membership, Organization, Role
from app.models.query_log import QueryLog
from app.models.user import User

__all__ = [
    "Base",
    "CacheIndex",
    "Chunk",
    "Citation",
    "Document",
    "Membership",
    "Organization",
    "QueryLog",
    "Role",
    "User",
]
```

`backend/app/db.py`:
```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings
from app.models.base import Base

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker | None = None


def _get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    _get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


def get_session():
    return get_sessionmaker()()


async def init_db() -> None:
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

Alembic migration `backend/alembic/versions/0001_initial.py` — create all tables via
`Base.metadata.create_all` (run with `alembic upgrade head`). Configure `backend/alembic.ini`
(sqlalchemy.url from env) and `backend/alembic/env.py` to import `app.models` so metadata is
populated. For the reference, `init_db()` (create_all) is used in tests; the migration exists for
real environments.
Alembic migration `backend/alembic/versions/0002_add_pending_version.py` — adds
`documents.pending_version` (nullable Integer). Because 0001 is create_all-driven (fresh
databases already get the column via metadata), 0002 inspects existing columns and skips if
`pending_version` is present, making `upgrade head` idempotent in both fresh and upgraded
databases; the guarded `downgrade` drops the column only if present.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_models.py -v`
Expected: PASS. (Requires Postgres running; `DATABASE_URL` defaults to `localhost:5432`.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/models backend/app/db.py backend/alembic backend/tests/test_models.py
git commit -m "feat(backend): add SQLAlchemy models and async db session"
```

---

### Task 1.2: MinIO (S3) client

**Files:**
- Create: `backend/app/core/s3.py`
- Test: `backend/tests/test_s3.py`

**Interfaces:**
- Consumes: `get_settings()`, `StorageError`.
- Produces in `backend/app/core/s3.py`:
  - `put_object(org_id, user_id, doc_id, filename: str, data: bytes) -> str` (s3 key) — creates bucket
    if missing, enables versioning; stores original filename in metadata; returns s3 key.
  - `get_object(s3_key: str) -> bytes`.
  - `delete_object(s3_key: str) -> None`.
  - `key_to_parts(s3_key: str) -> tuple[str, str, str, str]` (org_id, user_id, doc_id, uuid).
  - `build_key(org_id, user_id, doc_id, uuid_ext: str) -> str`.

- [ ] **Step 1: Write the failing s3 test**

`backend/tests/test_s3.py`:
```python
import pytest

from app.core.s3 import (
    build_key,
    delete_object,
    get_object,
    key_to_parts,
    put_object,
)


def test_build_key_shape():
    key = build_key("org1", "user1", "doc1", "abc123.txt")
    assert key == "documents/org1/user1/doc1/abc123.txt"


def test_key_to_parts():
    org, user, doc, name = key_to_parts("documents/o/u/d/x.txt")
    assert (org, user, doc, name) == ("o", "u", "d", "x.txt")


@pytest.mark.integration
async def test_put_get_delete_roundtrip():
    key = put_object("org-i", "user-i", "doc-i", "hello.txt", b"hello world")
    assert (await get_object(key)) == b"hello world"
    delete_object(key)
    with pytest.raises(Exception):
        await get_object(key)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_s3.py -v`
Expected: FAIL — `ModuleNotFoundError: app.core.s3`.

- [ ] **Step 3: Write implementation**

`backend/app/core/s3.py`:
```python
import asyncio
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.errors import StorageError


def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
    )


def _ensure_bucket(session, bucket: str) -> None:
    try:
        session.head_bucket(Bucket=bucket)
    except ClientError:
        session.create_bucket(Bucket=bucket)
        session.put_bucket_versioning(
            Bucket=bucket, VersioningConfiguration={"Status": "Enabled"}
        )


def build_key(org_id: str, user_id: str, doc_id: str, uuid_ext: str) -> str:
    return f"documents/{org_id}/{user_id}/{doc_id}/{uuid_ext}"


def key_to_parts(s3_key: str) -> tuple[str, str, str, str]:
    _bucket, org_id, user_id, doc_id, name = s3_key.split("/", 4)
    return org_id, user_id, doc_id, name


def put_object(org_id: str, user_id: str, doc_id: str, filename: str, data: bytes) -> str:
    session = _client()
    s = get_settings()
    _ensure_bucket(session, s.s3_bucket)
    key = build_key(org_id, user_id, doc_id, f"{uuid4()}.{filename.split('.')[-1]}")
    try:
        session.put_object(
            Bucket=s.s3_bucket,
            Key=key,
            Body=data,
            Metadata={"original_filename": filename},
        )
    except ClientError as exc:
        raise StorageError(detail=f"s3 put failed: {exc}") from exc
    return key


def get_object(s3_key: str) -> bytes:
    session = _client()
    s = get_settings()
    try:
        resp = session.get_object(Bucket=s.s3_bucket, Key=s3_key)
        return resp["Body"].read()
    except ClientError as exc:
        raise StorageError(detail=f"s3 get failed: {exc}") from exc


def delete_object(s3_key: str) -> None:
    session = _client()
    s = get_settings()
    try:
        session.delete_object(Bucket=s.s3_bucket, Key=s3_key)
    except ClientError as exc:
        raise StorageError(detail=f"s3 delete failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_s3.py -v`
Expected: PASS (unit tests; the `@pytest.mark.integration` test is skipped unless `-m integration`
and requires MinIO running).

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/s3.py backend/tests/test_s3.py
git commit -m "feat(backend): add MinIO S3 client with versioning and key scheme"
```

---

### Task 1.3: Qdrant client + collection setup

**Files:**
- Create: `backend/app/core/qdrant_store.py`
- Test: `backend/tests/test_qdrant_store.py`

**Interfaces:**
- Consumes: `get_settings()`, `StorageError`.
- Produces in `backend/app/core/qdrant_store.py`:
  - `ensure_collection() -> None` — creates collection `chunks` with named dense vector
    `dense_vector` (size `embed_dim`, `Distance.COSINE`) and sparse config `bm25_sparse_vector` with
    `Modifier.IDF`.
  - `upsert_point(point_id: str, dense: list[float], sparse_indices: list[int],
    sparse_values: list[float], payload: dict) -> None`
  - `upsert_points_batch(points: list[dict]) -> None`
  - `hybrid_search(dense: list[float], sparse_indices: list[int], sparse_values: list[float],
    payload_filter: dict, limit: int) -> list[dict]` — runs prefetch dense + sparse with the same
    filter, RRF fusion, returns points with payload + score.
  - `delete_points(point_ids: list[str], payload_filter: dict | None = None) -> None`
  - `build_payload_filter(org_id: str, user_ids: list[str] | None, extra: dict) -> dict`

- [ ] **Step 1: Write the failing qdrant test**

`backend/tests/test_qdrant_store.py`:
```python
import pytest

from app.core.qdrant_store import build_payload_filter, ensure_collection


def test_build_payload_filter_org_scoped():
    f = build_payload_filter(org_id="org1", user_ids=["u1", "u2"], extra={"year": 2025})
    assert f is not None


@pytest.mark.integration
def test_ensure_collection_runs():
    ensure_collection()
    assert True
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_qdrant_store.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/core/qdrant_store.py`:
```python
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.core.errors import StorageError


def _client() -> AsyncQdrantClient:
    s = get_settings()
    return AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)


async def ensure_collection() -> None:
    s = get_settings()
    client = _client()
    try:
        exists = await client.collection_exists(s.qdrant_collection)
        if not exists:
            await client.create_collection(
                collection_name=s.qdrant_collection,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=s.embed_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse_vector": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
    except Exception as exc:
        raise StorageError(detail=f"qdrant ensure_collection failed: {exc}") from exc
    finally:
        await client.close()


async def upsert_points_batch(points: list[dict]) -> None:
    s = get_settings()
    client = _client()
    try:
        structs = [
            models.PointStruct(
                id=p["id"],
                vector={
                    "dense_vector": p["dense"],
                    "bm25_sparse_vector": models.SparseVector(
                        indices=p["sparse_indices"], values=p["sparse_values"]
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ]
        await client.upsert(collection_name=s.qdrant_collection, points=structs)
    except Exception as exc:
        raise StorageError(detail=f"qdrant upsert failed: {exc}") from exc
    finally:
        await client.close()


async def upsert_point(
    point_id: str,
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload: dict,
) -> None:
    await upsert_points_batch(
        [
            {
                "id": point_id,
                "dense": dense,
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values,
                "payload": payload,
            }
        ]
    )


async def hybrid_search(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload_filter: dict,
    limit: int = 10,
) -> list[dict]:
    s = get_settings()
    client = _client()
    try:
        result = await client.query_points(
            collection_name=s.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense,
                    using="dense_vector",
                    limit=limit * 2,
                    filter=models.Filter(**payload_filter) if payload_filter else None,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                    using="bm25_sparse_vector",
                    limit=limit * 2,
                    filter=models.Filter(**payload_filter) if payload_filter else None,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [point.model_dump() for point in result.points]
    except Exception as exc:
        raise StorageError(detail=f"qdrant search failed: {exc}") from exc
    finally:
        await client.close()


async def delete_points(point_ids: list[str], payload_filter: dict | None = None) -> None:
    s = get_settings()
    client = _client()
    try:
        if point_ids:
            await client.delete(collection_name=s.qdrant_collection, points_selector=point_ids)
        elif payload_filter:
            await client.delete(
                collection_name=s.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(**payload_filter)
                ),
            )
    except Exception as exc:
        raise StorageError(detail=f"qdrant delete failed: {exc}") from exc
    finally:
        await client.close()


def build_payload_filter(org_id: str, user_ids: list[str] | None, extra: dict) -> dict:
    must: list[dict] = [{"key": "org_id", "match": {"value": org_id}}]
    if user_ids:
        must.append({"key": "user_id", "match": {"any": user_ids}})
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            must.append({"key": k, "match": {"any": list(v)}})
        else:
            must.append({"key": k, "match": {"value": v}})
    return {"must": must}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_qdrant_store.py -v`
Expected: PASS (unit; integration requires Qdrant running).

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/qdrant_store.py backend/tests/test_qdrant_store.py
git commit -m "feat(backend): add Qdrant client with dense+sparse collection and hybrid search"
```

---

### Task 1.4: Redis clients (semantic cache + rate limit)

**Files:**
- Create: `backend/app/core/redis_store.py`
- Test: `backend/tests/test_redis_store.py`

**Interfaces:**
- Consumes: `get_settings()`.
- Produces in `backend/app/core/redis_store.py`:
  - `class RedisCache`:
    - `async set(key: str, value: dict, ttl: int | None = None) -> None` (JSON-encoded)
    - `async get(key: str) -> dict | None`
    - `async delete(key: str) -> None`
    - `async delete_by_pattern(pattern: str) -> int`
  - `class RateLimiter`:
    - `async allow(user_key: str, limit: int, window_seconds: int) -> bool`
  - module-level `get_cache() -> RedisCache` and `get_limiter() -> RateLimiter` (singletons).

- [ ] **Step 1: Write the failing redis test**

`backend/tests/test_redis_store.py`:
```python
import pytest

from app.core.redis_store import get_cache, get_limiter


@pytest.mark.integration
async def test_cache_set_get_delete():
    cache = get_cache()
    await cache.set("k", {"a": 1})
    assert (await cache.get("k")) == {"a": 1}
    await cache.delete("k")
    assert (await cache.get("k")) is None


@pytest.mark.integration
async def test_rate_limiter_allows_then_blocks():
    limiter = get_limiter()
    assert await limiter.allow("u1", limit=2, window_seconds=60)
    assert await limiter.allow("u1", limit=2, window_seconds=60)
    assert not await limiter.allow("u1", limit=2, window_seconds=60)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_redis_store.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/core/redis_store.py`:
```python
import json
import time

import redis.asyncio as aioredis

from app.core.config import get_settings


class RedisCache:
    def __init__(self, url: str) -> None:
        self._r = aioredis.from_url(url, decode_responses=True)

    async def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        await self._r.set(key, json.dumps(value), ex=ttl)

    async def get(self, key: str) -> dict | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw else None

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def delete_by_pattern(self, pattern: str) -> int:
        keys = [k async for k in self._r.scan_iter(match=pattern)]
        if keys:
            return await self._r.delete(*keys)
        return 0

    async def sadd(self, key: str, *members: str) -> None:
        await self._r.sadd(key, *members)

    async def smembers(self, key: str) -> set[str]:
        return set(await self._r.smembers(key))


class RateLimiter:
    def __init__(self, url: str) -> None:
        self._r = aioredis.from_url(url, decode_responses=True)

    async def allow(self, user_key: str, limit: int, window_seconds: int) -> bool:
        now = int(time.time())
        window = now // window_seconds
        k = f"rl:{user_key}:{window}"
        count = await self._r.incr(k)
        if count == 1:
            await self._r.expire(k, window_seconds)
        return count <= limit


_cache: RedisCache | None = None
_limiter: RateLimiter | None = None


def get_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache(get_settings().redis_url)
    return _cache


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().redis_url)
    return _limiter
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_redis_store.py -v`
Expected: PASS (integration; requires Redis running).

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/redis_store.py backend/tests/test_redis_store.py
git commit -m "feat(backend): add Redis cache and rate limiter clients"
```

---

**Phase 1 exit check:** all unit tests green; integration tests pass with docker-compose services up
(`docker compose -f infra/docker-compose.yml up -d`).