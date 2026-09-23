import hashlib

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import app.db as db_module
from app.api.routes_documents import MAX_UPLOAD_BYTES
from app.auth.security import create_access_token
from app.db import init_db
from app.main import create_app
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.organization import Organization
from app.models.user import User


def test_max_upload_constant():
    assert MAX_UPLOAD_BYTES == 50 * 1024 * 1024


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _seed(*rows) -> None:
    async with db_module.get_session() as session:
        for row in rows:
            session.add(row)
        await session.commit()


def _org(org_id: str) -> Organization:
    return Organization(id=org_id, name=f"org-{org_id}")


def _user(user_id: str) -> User:
    return User(id=user_id, email=f"{user_id}@example.com", password_hash="x")


def _doc(
    doc_id: str,
    org_id: str,
    user_id: str,
    status: DocumentStatus = DocumentStatus.EMBEDDED,
    version: int = 1,
    content_hash: str = "0" * 64,
) -> Document:
    return Document(
        id=doc_id,
        user_id=user_id,
        org_id=org_id,
        original_filename=f"{doc_id}.txt",
        status=status,
        content_hash=content_hash,
        current_version=version,
    )


def _chunk(
    chunk_id: str,
    doc_id: str,
    org_id: str,
    user_id: str,
    version: int,
    start_offset: int,
    end_offset: int,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        user_id=user_id,
        org_id=org_id,
        chunk_text=f"chunk {chunk_id}",
        start_offset=start_offset,
        end_offset=end_offset,
        version=version,
    )


class _NoS3:
    calls: list = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("put_object must not be called")


class _NoPublish:
    events: list = []

    async def __call__(self, *args, **kwargs):
        self.events.append((args, kwargs))
        raise AssertionError("publish_ingestion must not be called")


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/routes_documents.db")
    db_module._engine = None
    db_module._sessionmaker = None
    await init_db()
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    if db_module._engine is not None:
        await db_module._engine.dispose()
    db_module._engine = None
    db_module._sessionmaker = None


async def test_upload_creates_pending_document_and_publishes(client, monkeypatch):
    await _seed(_org("org-1"), _user("user-1"))
    s3_calls: list = []
    events: list = []

    def fake_put_object(org_id, user_id, doc_id, filename, data):
        s3_calls.append((org_id, user_id, doc_id, filename, data))
        return f"documents/{org_id}/{user_id}/{doc_id}/uuid.txt"

    async def fake_publish(doc_id, s3_key):
        events.append((doc_id, s3_key))

    monkeypatch.setattr("app.api.routes_documents.put_object", fake_put_object)
    monkeypatch.setattr("app.api.routes_documents.publish_ingestion", fake_publish)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        headers=_auth(token),
    )

    assert resp.status_code == 200
    body = resp.json()
    doc_id = body["doc_id"]
    assert body["status"] == "processing"
    assert body["s3_key"] == f"documents/org-1/user-1/{doc_id}/uuid.txt"

    # fresh PENDING row with no pending_version before the ingest message is published
    async with db_module.get_session() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        assert doc.org_id == "org-1"
        assert doc.user_id == "user-1"
        assert doc.status == DocumentStatus.PENDING
        assert doc.pending_version is None
        assert doc.content_hash == hashlib.sha256(b"hello world").hexdigest()
        assert doc.current_version == 1

    assert s3_calls == [("org-1", "user-1", doc_id, "notes.txt", b"hello world")]
    assert events == [(doc_id, f"documents/org-1/user-1/{doc_id}/uuid.txt")]


async def test_upload_rejects_file_over_limit(client, monkeypatch):
    await _seed(_org("org-1"), _user("user-1"))
    events: list = []

    async def fake_publish(doc_id, s3_key):
        events.append((doc_id, s3_key))

    monkeypatch.setattr("app.api.routes_documents.put_object", _NoS3())
    monkeypatch.setattr("app.api.routes_documents.publish_ingestion", fake_publish)

    token = create_access_token(sub="user-1", org_id="org-1")
    big = b"x" * (MAX_UPLOAD_BYTES + 1)
    resp = await client.post(
        "/documents/upload",
        files={"file": ("big.bin", big, "application/octet-stream")},
        headers=_auth(token),
    )

    assert resp.status_code == 422
    assert resp.json()["error"] == "file exceeds 50MB limit"
    assert events == []
    async with db_module.get_session() as session:
        rows = (await session.execute(select(Document))).scalars().all()
        assert rows == []


async def test_upload_duplicate_embedded_returns_early(client, monkeypatch):
    content_hash = hashlib.sha256(b"hello").hexdigest()
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-dup", "org-1", "user-1", DocumentStatus.EMBEDDED, content_hash=content_hash),
    )
    monkeypatch.setattr("app.api.routes_documents.put_object", _NoS3())
    monkeypatch.setattr("app.api.routes_documents.publish_ingestion", _NoPublish())

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post(
        "/documents/upload",
        files={"file": ("hello.txt", b"hello", "text/plain")},
        headers=_auth(token),
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "doc_id": "doc-dup",
        "status": "duplicate",
        "already_embedded": True,
    }
    async with db_module.get_session() as session:
        rows = (
            (await session.execute(select(Document).where(Document.content_hash == content_hash)))
            .scalars()
            .all()
        )
        assert [d.id for d in rows] == ["doc-dup"]


async def test_upload_same_hash_other_org_is_not_duplicate(client, monkeypatch):
    content_hash = hashlib.sha256(b"hello").hexdigest()
    await _seed(
        _org("org-1"),
        _org("org-2"),
        _user("user-1"),
        _user("user-2"),
        _doc(
            "doc-other-org", "org-2", "user-2", DocumentStatus.EMBEDDED, content_hash=content_hash
        ),
    )
    s3_calls: list = []
    events: list = []

    def fake_put_object(org_id, user_id, doc_id, filename, data):
        s3_calls.append((org_id, user_id, doc_id, filename, data))
        return f"documents/{org_id}/{user_id}/{doc_id}/uuid.txt"

    async def fake_publish(doc_id, s3_key):
        events.append((doc_id, s3_key))

    monkeypatch.setattr("app.api.routes_documents.put_object", fake_put_object)
    monkeypatch.setattr("app.api.routes_documents.publish_ingestion", fake_publish)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post(
        "/documents/upload",
        files={"file": ("hello.txt", b"hello", "text/plain")},
        headers=_auth(token),
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "processing"
    assert body["doc_id"] != "doc-other-org"
    assert len(events) == 1
    async with db_module.get_session() as session:
        doc = await session.get(Document, body["doc_id"])
        assert doc is not None
        assert doc.org_id == "org-1"


async def test_upload_requires_auth(client):
    resp = await client.post("/documents/upload", files={"file": ("a.txt", b"x", "text/plain")})
    assert resp.status_code == 401


async def test_list_documents_scoped_to_caller(client):
    await _seed(
        _org("org-1"),
        _org("org-2"),
        _user("user-1"),
        _user("user-2"),
        _doc("doc-own", "org-1", "user-1"),
        _doc("doc-colleague", "org-1", "user-2"),
        _doc("doc-other-org", "org-2", "user-1"),
    )

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents", headers=_auth(token))

    assert resp.status_code == 200
    body = resp.json()
    assert [d["id"] for d in body] == ["doc-own"]
    assert body[0]["filename"] == "doc-own.txt"
    assert body[0]["status"] == "EMBEDDED"
    assert body[0]["version"] == 1


async def test_list_documents_requires_auth(client):
    resp = await client.get("/documents")
    assert resp.status_code == 401


async def test_get_document_returns_status_and_version(client):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-1", "org-1", "user-1", DocumentStatus.PROCESSING, version=3),
    )

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents/doc-1", headers=_auth(token))

    assert resp.status_code == 200
    assert resp.json() == {
        "id": "doc-1",
        "filename": "doc-1.txt",
        "status": "PROCESSING",
        "version": 3,
    }


async def test_get_document_other_org_hidden(client):
    await _seed(_org("org-2"), _user("user-2"), _doc("doc-2", "org-2", "user-2"))

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents/doc-2", headers=_auth(token))

    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"


async def test_get_document_missing_returns_422(client):
    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents/never-seen", headers=_auth(token))
    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"


async def test_delete_document_cancels_and_invalidates(client, monkeypatch):
    await _seed(_org("org-1"), _user("user-1"), _doc("doc-1", "org-1", "user-1"))
    cancel_calls: list = []
    invalidate_calls: list = []

    async def fake_cancel(doc_id):
        cancel_calls.append(doc_id)

    async def fake_invalidate(org_id, doc_id):
        invalidate_calls.append((org_id, doc_id))

    monkeypatch.setattr("app.ingestion.cancel.cancel_document", fake_cancel)
    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", fake_invalidate)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.delete("/documents/doc-1", headers=_auth(token))

    assert resp.status_code == 200
    assert resp.json() == {"doc_id": "doc-1", "status": "cancelled"}
    assert cancel_calls == ["doc-1"]
    assert invalidate_calls == [("org-1", "doc-1")]


async def test_delete_document_other_org_hidden_and_untouched(client, monkeypatch):
    await _seed(_org("org-2"), _user("user-2"), _doc("doc-2", "org-2", "user-2"))

    async def must_not_cancel(doc_id):
        raise AssertionError("cancel_document must not be called")

    async def must_not_invalidate(org_id, doc_id):
        raise AssertionError("invalidate_for_doc must not be called")

    monkeypatch.setattr("app.ingestion.cancel.cancel_document", must_not_cancel)
    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", must_not_invalidate)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.delete("/documents/doc-2", headers=_auth(token))

    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"


async def test_doc_chunks_returns_current_version_only(client):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-1", "org-1", "user-1", DocumentStatus.EMBEDDED, version=2),
        _doc("doc-2", "org-1", "user-1", DocumentStatus.EMBEDDED, version=1),
        _chunk("chunk-v1", "doc-1", "org-1", "user-1", version=1, start_offset=0, end_offset=5),
        _chunk("chunk-v2a", "doc-1", "org-1", "user-1", version=2, start_offset=0, end_offset=5),
        _chunk("chunk-v2b", "doc-1", "org-1", "user-1", version=2, start_offset=6, end_offset=11),
        _chunk(
            "chunk-other-doc", "doc-2", "org-1", "user-1", version=1, start_offset=0, end_offset=4
        ),
    )

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents/doc-1/chunks", headers=_auth(token))

    assert resp.status_code == 200
    body = sorted(resp.json(), key=lambda c: c["start_offset"])
    assert body == [
        {"id": "chunk-v2a", "page": 0, "start_offset": 0, "end_offset": 5},
        {"id": "chunk-v2b", "page": 0, "start_offset": 6, "end_offset": 11},
    ]


async def test_doc_chunks_other_org_doc_hidden(client):
    await _seed(_org("org-2"), _user("user-2"), _doc("doc-2", "org-2", "user-2"))

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents/doc-2/chunks", headers=_auth(token))

    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"


async def test_doc_chunks_missing_doc_returns_422(client):
    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents/never-seen/chunks", headers=_auth(token))
    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"
