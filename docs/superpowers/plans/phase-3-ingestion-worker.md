# Phase 3 — Ingestion Worker (Write Path)

**Goal:** Implement document parsing (PDF, DOCX, MD, TXT), the ingestion pipeline
(parse → clean → chunk → embed → store → status), the RabbitMQ consumer driven by MinIO object
events, document versioning with atomic `current_version` flip, a periodic cleanup job for stale
versions, and upload cancellation with cleanup.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md` (sections 4, 5, 9)

## Dependencies

Phase 0, 1, 2 (chunkers, embeddings, qdrant_store, s3, db, models, errors).

---

### Task 3.1: Document parsers

**Files:**
- Create: `backend/app/rag/ingestion/__init__.py`
- Create: `backend/app/rag/ingestion/parsers.py`
- Test: `backend/tests/test_parsers.py`

**Interfaces:**
- Produces in `backend/app/rag/ingestion/parsers.py`:
  - `@dataclass ParsedPage`: `text: str`, `page_number: int`.
  - `@dataclass ParsedDocument`: `pages: list[ParsedPage]`, `full_text: str`.
  - `parse_pdf(data: bytes) -> ParsedDocument` — PyMuPDF; one ParsedPage per page.
  - `parse_docx(data: bytes) -> ParsedDocument` — python-docx; paragraphs joined, page 0.
  - `parse_markdown(data: bytes) -> ParsedDocument` — decode text as-is, page 0.
  - `parse_txt(data: bytes) -> ParsedDocument` — decode + normalize, page 0.
  - `parse_bytes(filename: str, data: bytes) -> ParsedDocument` — dispatches by extension
    (`.pdf`, `.docx`, `.md`, `.txt`); raises `IngestionError` on unknown type.

- [ ] **Step 1: Write the failing parser test**

`backend/tests/test_parsers.py`:
```python
import pytest

from app.rag.ingestion.parsers import (
    ParsedDocument,
    parse_bytes,
    parse_markdown,
    parse_txt,
)


def test_parse_txt():
    doc = parse_txt(b"hello\nworld")
    assert doc.full_text == "hello\nworld"
    assert len(doc.pages) == 1


def test_parse_markdown():
    doc = parse_markdown(b"# Title\n\nbody")
    assert "# Title" in doc.full_text


def test_parse_bytes_unknown_type():
    with pytest.raises(Exception):
        parse_bytes("file.xyz", b"data")


def test_parse_bytes_txt():
    doc = parse_bytes("notes.txt", b"some text")
    assert isinstance(doc, ParsedDocument)
    assert "some text" in doc.full_text
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_parsers.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/rag/ingestion/parsers.py`:
```python
from dataclasses import dataclass

from app.core.errors import IngestionError


@dataclass
class ParsedPage:
    text: str
    page_number: int


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    full_text: str


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise IngestionError(detail="PyMuPDF not installed") from exc
    pages: list[ParsedPage] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            pages.append(ParsedPage(text=page.get_text(), page_number=i))
    return ParsedDocument(pages=pages, full_text="\n".join(p.text for p in pages))


def parse_docx(data: bytes) -> ParsedDocument:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise IngestionError(detail="python-docx not installed") from exc
    import io

    doc = DocxDocument(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    return ParsedDocument(pages=[ParsedPage(text=text, page_number=0)], full_text=text)


def parse_markdown(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    return ParsedDocument(pages=[ParsedPage(text=text, page_number=0)], full_text=text)


def parse_txt(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return ParsedDocument(pages=[ParsedPage(text=text, page_number=0)], full_text=text)


def parse_bytes(filename: str, data: bytes) -> ParsedDocument:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return parse_pdf(data)
    if ext == "docx":
        return parse_docx(data)
    if ext == "md":
        return parse_markdown(data)
    if ext == "txt":
        return parse_txt(data)
    raise IngestionError(detail=f"unsupported file type: {ext or 'unknown'}")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_parsers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/ingestion backend/tests/test_parsers.py
git commit -m "feat(ingestion): add PDF/DOCX/MD/TXT parsers"
```

---

### Task 3.2: Ingestion pipeline orchestrator

**Files:**
- Create: `backend/app/rag/ingestion/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `parse_bytes`, `get_chunker`, `ChunkData`, `dense_embed`, `sparse_embed`,
  `upsert_points_batch`, `Document`/`Chunk` models, `get_session`.
- Produces in `backend/app/rag/ingestion/pipeline.py`:
  - `async ingest_document(doc_id: str, s3_key: str) -> int` — full pipeline for ONE document at its
    CURRENT version; returns the new chunk count. Steps:
    1. Load Document row; mark `PROCESSING`.
    2. Fetch raw bytes from S3.
    3. Dedup via `content_hash`; if unchanged, mark `EMBEDDED` and return 0.
    4. Parse → chunk each page → enrich metadata (doc_id, user_id, org_id, page, offsets, version).
    5. Embed dense + sparse (batching), upsert to Qdrant with payload.
    6. Insert Chunk rows (version = current_version) into Postgres.
    7. Mark `EMBEDDED`; commit. Return chunk count.
  - `async ingest_versioned(doc_id: str, s3_key: str, new_version: int) -> int` — writes chunks with
    `version = new_version`, then on success flips `documents.current_version` and invalidates cache.
  - `async cleanup_stale(doc_id: str) -> int` — deletes chunks/vectors with `version < current_version`.

- [ ] **Step 1: Write the failing pipeline test**

`backend/tests/test_pipeline.py`:
```python
import pytest

from app.rag.ingestion.pipeline import ingest_document


@pytest.mark.integration
async def test_ingest_document_returns_chunk_count():
    # requires running MinIO, Postgres, Qdrant and a pre-seeded Document row
    count = await ingest_document(
        "doc-id-1", "documents/org/user/doc-id-1/sample.txt"
    )
    assert count >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/rag/ingestion/pipeline.py`:
```python
import asyncio
import hashlib
import logging
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import IngestionError
from app.core.qdrant_store import delete_points, upsert_points_batch
from app.core.s3 import get_object, put_object
from app.db import get_session
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.chunkers.text import get_chunker
from app.rag.embed.embeddings import dense_embed, sparse_embed
from app.rag.ingestion.parsers import parse_bytes

logger = logging.getLogger(__name__)
EMBED_BATCH = 32


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _embed_and_store(doc: Document, chunks: list[dict]) -> None:
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        texts = [c["chunk_text"] for c in batch]
        dense = await dense_embed(texts)
        sparse = sparse_embed(texts)
        points = []
        for c, d, s in zip(batch, dense, sparse):
            points.append(
                {
                    "id": c["id"],
                    "dense": d,
                    "sparse_indices": s.indices,
                    "sparse_values": s.values,
                    "payload": {
                        "doc_id": doc.id,
                        "user_id": doc.user_id,
                        "org_id": doc.org_id,
                        "chunk_text_hash": hashlib.sha256(
                            c["chunk_text"].encode()
                        ).hexdigest(),
                        "version": c["version"],
                        "page_number": c["page_number"],
                        "chunk_index": c["chunk_index"],
                        "word_count": len(c["chunk_text"].split()),
                        "char_count": len(c["chunk_text"]),
                        "type": doc.original_filename.rsplit(".", 1)[-1].lower(),
                    },
                }
            )
        await upsert_points_batch(points)


async def ingest_document(doc_id: str, s3_key: str) -> int:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise IngestionError(detail=f"document not found: {doc_id}")
        doc.status = DocumentStatus.PROCESSING
        await session.commit()

        # NOTE (failure path, as implemented in app/rag/ingestion/pipeline.py):
        # everything below runs inside try/except; on any exception the document
        # is marked FAILED and, if any batch was already upserted, Qdrant points
        # for THIS (doc_id, version) are best-effort deleted so partial batches
        # never leave orphaned, searchable duplicates:
        #     delete_points([], {"must": [
        #         {"key": "doc_id", "match": {"value": doc_id}},
        #         {"key": "version", "match": {"value": chunk_version}},
        #     ]})
        # wrapped in try/except (cleanup failure is logged, never raised) so it
        # cannot mask the original error.

        raw = get_object(s3_key)
        content_hash = _hash(raw)
        if content_hash == doc.content_hash:
            doc.status = DocumentStatus.EMBEDDED
            await session.commit()
            return 0

        parsed = parse_bytes(doc.original_filename, raw)
        chunker = get_chunker(doc.original_filename.rsplit(".", 1)[-1])
        chunk_data = []
        for page in parsed.pages:
            chunk_data.extend(chunker.chunk(page.text, page_number=page.page_number))

        from uuid import uuid4

        chunks: list[dict] = []
        for cd in chunk_data:
            chunks.append(
                {
                    "id": str(uuid4()),
                    "doc_id": doc.id,
                    "user_id": doc.user_id,
                    "org_id": doc.org_id,
                    "chunk_text": cd.text,
                    "page_number": cd.page_number,
                    "start_offset": cd.start_offset,
                    "end_offset": cd.end_offset,
                    "version": doc.current_version,
                    "chunk_index": cd.chunk_index,
                }
            )

        await _embed_and_store(doc, chunks)

        session.add_all(
            Chunk(**{k: c[k] for k in ("id", "doc_id", "user_id", "org_id", "chunk_text",
                "page_number", "start_offset", "end_offset", "version")})
            for c in chunks
        )
        doc.content_hash = content_hash
        doc.status = DocumentStatus.EMBEDDED
        await session.commit()
        return len(chunks)


async def ingest_versioned(doc_id: str, s3_key: str, new_version: int) -> int:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise IngestionError(detail=f"document not found: {doc_id}")
        old_version = doc.current_version
        doc.current_version = new_version
        await session.commit()
        try:
            count = await ingest_document(doc_id, s3_key)
        except Exception:
            doc.current_version = old_version
            await session.commit()
            raise
        return count


async def cleanup_stale(doc_id: str) -> int:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return 0
        stale = (
            await session.execute(
                select(Chunk.id).where(
                    Chunk.doc_id == doc_id, Chunk.version < doc.current_version
                )
            )
        ).scalars().all()
        if stale:
            await delete_points(list(stale), {"must": [{"key": "doc_id", "match": {"value": doc_id}}]})
        result = await session.execute(
            select(Chunk).where(
                Chunk.doc_id == doc_id, Chunk.version < doc.current_version
            )
        )
        for c in result.scalars():
            await session.delete(c)
        await session.commit()
        return len(stale)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_pipeline.py -v`
Expected: PASS (integration; requires MinIO, Postgres, Qdrant running and a seeded Document row).

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/ingestion/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat(ingestion): add ingestion pipeline with versioning and stale cleanup"
```

---

### Task 3.3: RabbitMQ consumer + MinIO event wiring

**Files:**
- Create: `backend/app/ingestion/__init__.py`
- Create: `backend/app/ingestion/consumer.py`
- Create: `backend/app/ingestion/worker.py`
- Test: `backend/tests/test_consumer.py`

**Interfaces:**
- Consumes: `get_settings`, `get_session`, `ingest_document`, `Document` model.
- Produces:
  - `declare_queues(channel) -> None` — declares `ingestion` (durable, DLX `ingestion.dlx` →
    `ingestion.dlq`, no TTL on the work queue), `ingestion.retry` (durable, `x-message-ttl: 30000`,
    dead-letters back to `ingestion` via the default exchange), and binds the DLQ to the DLX.
  - `async process_message(body: dict) -> None` — looks up Document by id, calls `ingest_document`,
    raises on failure (so the queue can retry/DLQ).
  - `async consume_loop() -> None` — connects via aio-pika, consumes `ingestion`, acks on success;
    on failure republishes to `ingestion.retry` (preserving `x-death` headers) until the
    `ingestion.retry` death count reaches `MAX_RETRIES = 3`, then rejects to DLQ.
  - `run_worker() -> None` — entrypoint (`python -m app.ingestion.worker`).

- [ ] **Step 1: Write the failing consumer test**

`backend/tests/test_consumer.py`:
```python
from app.ingestion.consumer import process_message


def test_process_message_signature():
    # unit-safety: no queue required for signature; integration behavior needs RabbitMQ
    assert callable(process_message)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_consumer.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/ingestion/consumer.py`:
```python
import json
import logging

from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage

from app.core.config import get_settings
from app.core.errors import IngestionError
from app.db import get_session
from app.models.document import Document, DocumentStatus
from app.rag.ingestion.pipeline import ingest_document

logger = logging.getLogger(__name__)
QUEUE = "ingestion"
DLQ = "ingestion.dlq"
RETRY = "ingestion.retry"
MAX_RETRIES = 3


async def declare_queues(channel: AbstractChannel) -> None:
    dlx = await channel.declare_exchange("ingestion.dlx", ExchangeType.DIRECT, durable=True)
    dlq = await channel.declare_queue(DLQ, durable=True, arguments={"x-queue-mode": "lazy"})
    await dlq.bind(dlx, routing_key=DLQ)
    await channel.declare_queue(
        QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "ingestion.dlx",
            "x-dead-letter-routing-key": DLQ,
            "x-max-length": 10000,
        },
    )
    await channel.declare_queue(
        RETRY,
        durable=True,
        arguments={
            "x-message-ttl": 30000,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": QUEUE,
        },
    )


declare = declare_queues


async def process_message(body: dict) -> None:
    doc_id = body.get("doc_id")
    s3_key = body.get("s3_key")
    if not doc_id or not s3_key:
        raise IngestionError(detail="message missing doc_id/s3_key")
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            raise IngestionError(detail=f"unknown doc {doc_id}")
        if doc.status == DocumentStatus.EMBEDDED:
            return
    await ingest_document(doc_id, s3_key)


def _retry_count(message: AbstractIncomingMessage) -> int:
    deaths = (message.headers or {}).get("x-death") or []
    for entry in deaths:
        if entry.get("queue") == RETRY:
            return int(entry.get("count", 0))
    return 0


async def consume_loop() -> None:
    settings = get_settings()
    connection = await connect_robust(settings.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=4)
    await declare_queues(channel)
    queue = await channel.get_queue(QUEUE)
    async with queue.iterator() as qiter:
        async for message in qiter:
            try:
                body = json.loads(message.body)
                await process_message(body)
                await message.ack()
            except Exception:
                retries = _retry_count(message)
                if retries >= MAX_RETRIES:
                    logger.exception(
                        "ingestion failed after retries, moving to DLQ",
                        extra={"body": message.body, "retries": retries},
                    )
                    await message.reject(requeue=False)
                else:
                    logger.exception(
                        "ingestion failed, scheduling retry",
                        extra={"body": message.body, "retries": retries},
                    )
                    await channel.default_exchange.publish(
                        Message(
                            body=message.body,
                            headers=dict(message.headers or {}),
                            delivery_mode=DeliveryMode.PERSISTENT,
                        ),
                        routing_key=RETRY,
                    )
                    await message.ack()
```

`backend/app/ingestion/worker.py`:
```python
import asyncio

from app.core.logging import setup_logging
from app.ingestion.consumer import consume_loop


async def main() -> None:
    setup_logging()
    await consume_loop()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_consumer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion backend/tests/test_consumer.py
git commit -m "feat(ingestion): add RabbitMQ consumer with DLQ and retry"
```

---

### Task 3.4: MinIO event → RabbitMQ publisher + cancellation + cleanup job

**Files:**
- Create: `backend/app/ingestion/publisher.py`
- Create: `backend/app/ingestion/cleanup_job.py`
- Create: `backend/app/ingestion/cancel.py`
- Test: `backend/tests/test_publisher.py`

**Interfaces:**
- Produces:
  - `async publish_ingestion(doc_id: str, s3_key: str) -> None` in `publisher.py` — publishes a JSON
    message `{doc_id, s3_key}` to the `ingestion` queue.
  - `async cancel_document(doc_id: str) -> None` in `cancel.py` — marks document `FAILED` (or
    `PENDING` on abort) and publishes a cancellation message the consumer checks; cleans partial rows.
  - `async cleanup_job_loop(interval_seconds: int = 3600) -> None` in `cleanup_job.py` — periodically
    finds docs with stale chunks and calls `cleanup_stale`; batches with retries.
  - `register_minio_webhook(app) -> None` — exposes `POST /internal/minio-event` that parses an
    S3-event-like payload and calls `publish_ingestion` (MinIO bucket notification → this endpoint).

- [ ] **Step 1: Write the failing publisher test**

`backend/tests/test_publisher.py`:
```python
from app.ingestion.publisher import publish_ingestion


def test_publish_ingestion_signature():
    assert callable(publish_ingestion)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_publisher.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/ingestion/publisher.py`:
```python
import json
import logging

from aio_pika import connect_robust

from app.core.config import get_settings
from app.ingestion.consumer import QUEUE, declare

logger = logging.getLogger(__name__)


async def publish_ingestion(doc_id: str, s3_key: str) -> None:
    settings = get_settings()
    connection = await connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        await declare(channel)  # declares ingestion, ingestion.retry (TTL dead-letters back), DLQ
        await channel.default_exchange.publish(
            json.dumps({"doc_id": doc_id, "s3_key": s3_key}).encode(),
            routing_key=QUEUE,
        )
    finally:
        await connection.close()
```

`backend/app/ingestion/cancel.py`:
```python
import logging

from app.db import get_session
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


async def cancel_document(doc_id: str) -> None:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None:
            return
        doc.status = DocumentStatus.FAILED
        await session.commit()
```

`backend/app/ingestion/cleanup_job.py`:
```python
import asyncio
import logging

from sqlalchemy import func, select

from app.db import get_session
from app.models.chunk import Chunk
from app.models.document import Document
from app.rag.ingestion.pipeline import cleanup_stale

logger = logging.getLogger(__name__)


async def find_stale_doc_ids(limit: int = 100) -> list[str]:
    async with get_session() as session:
        rows = await session.execute(
            select(Chunk.doc_id, func.max(Chunk.version).label("maxv"))
            .join(Document, Document.id == Chunk.doc_id)
            .group_by(Chunk.doc_id)
            .having(func.max(Chunk.version) < Document.current_version)
            .limit(limit)
        )
        return [r.doc_id for r in rows]


async def cleanup_job_loop(interval_seconds: int = 3600) -> None:
    while True:
        try:
            for doc_id in await find_stale_doc_ids():
                await cleanup_stale(doc_id)
        except Exception:
            logger.exception("cleanup job iteration failed")
        await asyncio.sleep(interval_seconds)
```

`backend/app/ingestion/minio_webhook.py`:
```python
import logging

from fastapi import APIRouter, Request

from app.core.errors import IngestionError
from app.ingestion.publisher import publish_ingestion

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/internal/minio-event")
async def minio_event(request: Request) -> dict:
    data = await request.json()
    for record in data.get("Records", []):
        key = record.get("s3", {}).get("object", {}).get("key", "")
        if not key:
            continue
        parts = key.split("/")
        # key: documents/<org>/<user>/<doc>/<uuid>.<ext>
        if len(parts) < 5:
            raise IngestionError(detail=f"unexpected s3 key: {key}")
        doc_id = parts[3]
        await publish_ingestion(doc_id, key)
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_publisher.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingestion/publisher.py backend/app/ingestion/cancel.py \
       backend/app/ingestion/cleanup_job.py backend/app/ingestion/minio_webhook.py \
       backend/tests/test_publisher.py
git commit -m "feat(ingestion): add publisher, cancellation, cleanup job, and minio webhook"
```

---

**Phase 3 exit check:** unit tests green; integration: upload a doc to MinIO → event → worker
ingests → chunks present in Postgres and vectors in Qdrant, `documents.status = EMBEDDED`.