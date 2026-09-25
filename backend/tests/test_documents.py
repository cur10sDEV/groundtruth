import sqlite3

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import app.api.routes_documents as routes_documents
import app.db as db_module
import app.ingestion.cancel as cancel_module
from app.auth.security import create_access_token
from app.db import init_db
from app.main import create_app
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.organization import Organization
from app.models.user import User


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
    failure_reason: str | None = None,
) -> Document:
    return Document(
        id=doc_id,
        user_id=user_id,
        org_id=org_id,
        original_filename=f"{doc_id}.txt",
        status=status,
        content_hash=content_hash,
        current_version=version,
        failure_reason=failure_reason,
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


async def test_sign_new_document_creates_pending_row(client, monkeypatch):
    await _seed(_org("org-1"), _user("user-1"))
    presign_calls: list = []

    def fake_presign(org_id, user_id, doc_id, ext):
        presign_calls.append((org_id, user_id, doc_id, ext))
        return {
            "url": "u",
            "fields": {"key": "k"},
            "key": f"documents/{org_id}/{user_id}/{doc_id}/x.pdf",
        }

    monkeypatch.setattr("app.api.routes_documents.presign_upload", fake_presign)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post(
        "/documents/sign", json={"filename": "policy.pdf"}, headers=_auth(token)
    )

    assert resp.status_code == 200
    body = resp.json()
    doc_id = body["doc_id"]
    assert body["status"] == "pending"
    assert body["upload"] == {"url": "u", "fields": {"key": "k"}}
    assert "new_version" not in body

    async with db_module.get_session() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        assert doc.org_id == "org-1"
        assert doc.user_id == "user-1"
        assert doc.original_filename == "policy.pdf"
        assert doc.status == DocumentStatus.PENDING
        assert doc.pending_version is None
        assert doc.content_hash == ""  # unknown until the worker fetches the object
        assert doc.current_version == 1

    assert presign_calls == [("org-1", "user-1", doc_id, "pdf")]


async def test_sign_same_filename_embedded_doc_becomes_new_version(client, monkeypatch):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-v1", "org-1", "user-1", DocumentStatus.EMBEDDED, version=1),
    )
    presign_calls: list = []

    def fake_presign(org_id, user_id, doc_id, ext):
        presign_calls.append((org_id, user_id, doc_id, ext))
        return {
            "url": "u",
            "fields": {"key": "k"},
            "key": f"documents/{org_id}/{user_id}/{doc_id}/x.txt",
        }

    monkeypatch.setattr("app.api.routes_documents.presign_upload", fake_presign)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post(
        "/documents/sign", json={"filename": "doc-v1.txt"}, headers=_auth(token)
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-v1"
    assert body["status"] == "pending"
    assert body["new_version"] == 2
    assert body["upload"] == {"url": "u", "fields": {"key": "k"}}

    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-v1")
        assert doc.status == DocumentStatus.PENDING
        assert doc.pending_version == 2
        assert doc.current_version == 1
        rows = (await session.execute(select(Document))).scalars().all()
        assert [d.id for d in rows] == ["doc-v1"]  # no second row for a versioned re-upload

    assert presign_calls == [("org-1", "user-1", "doc-v1", "txt")]


def _flip_to_deleting(db_path, doc_id):
    # a second actor (the delete route) tombstones the row between /sign's
    # select and its status write; a sync connection is the only seam
    # available inside the statement-construct wrapper
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE documents SET status = 'DELETING' WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


async def test_sign_versioned_race_never_stomps_deleting(client, monkeypatch, tmp_path):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-v1", "org-1", "user-1", DocumentStatus.EMBEDDED, version=1),
    )
    real_update = routes_documents.update

    def flipping_update(*args, **kwargs):
        _flip_to_deleting(tmp_path / "routes_documents.db", "doc-v1")
        return real_update(*args, **kwargs)

    monkeypatch.setattr("app.api.routes_documents.update", flipping_update)
    monkeypatch.setattr(
        "app.api.routes_documents.presign_upload",
        lambda *a: {"url": "u", "fields": {"key": "k"}, "key": "k"},
    )

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post(
        "/documents/sign", json={"filename": "doc-v1.txt"}, headers=_auth(token)
    )

    assert resp.status_code == 200
    body = resp.json()
    # pinned behavior: the guarded update loses the race and /sign falls
    # through to a fresh document row instead of writing over the tombstone
    assert body["doc_id"] != "doc-v1"
    assert body["status"] == "pending"
    assert "new_version" not in body

    async with db_module.get_session() as session:
        old = await session.get(Document, "doc-v1")
        assert old is not None
        assert old.status == DocumentStatus.DELETING  # tombstone survived
        assert old.pending_version is None  # never overwritten with a PENDING bump
        assert old.current_version == 1
        rows = (await session.execute(select(Document))).scalars().all()
        fresh = [d for d in rows if d.id != "doc-v1"]
        assert len(fresh) == 1
        assert fresh[0].status == DocumentStatus.PENDING
        assert fresh[0].pending_version is None
        assert fresh[0].current_version == 1


async def test_sign_requires_auth(client):
    resp = await client.post("/documents/sign", json={"filename": "a.txt"})
    assert resp.status_code == 401


@pytest.mark.parametrize("filename", ["../evil.txt", "sub\\dir.txt"])
async def test_sign_rejects_filenames_with_path_separators(client, filename):
    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.post("/documents/sign", json={"filename": filename}, headers=_auth(token))
    assert resp.status_code == 422


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


async def test_list_documents_excludes_deleting(client):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-embedded", "org-1", "user-1"),
        _doc("doc-tombstone", "org-1", "user-1", DocumentStatus.DELETING),
    )

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.get("/documents", headers=_auth(token))

    assert resp.status_code == 200
    assert [d["id"] for d in resp.json()] == ["doc-embedded"]


async def test_list_and_get_include_failure_reason(client):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-failed", "org-1", "user-1", DocumentStatus.FAILED, failure_reason="boom"),
    )

    token = create_access_token(sub="user-1", org_id="org-1")
    listed = await client.get("/documents", headers=_auth(token))
    assert listed.status_code == 200
    assert listed.json()[0]["failure_reason"] == "boom"

    got = await client.get("/documents/doc-failed", headers=_auth(token))
    assert got.status_code == 200
    assert got.json()["failure_reason"] == "boom"


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
        "failure_reason": None,
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


def _forbid_route_physical_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_delete_points(*args, **kwargs):
        raise AssertionError("route must not call delete_points")

    def must_not_delete_prefix(*args, **kwargs):
        raise AssertionError("route must not call delete_prefix")

    monkeypatch.setattr(
        "app.api.routes_documents.delete_points", must_not_delete_points, raising=False
    )
    monkeypatch.setattr(
        "app.api.routes_documents.delete_prefix", must_not_delete_prefix, raising=False
    )
    monkeypatch.setattr("app.core.qdrant_store.delete_points", must_not_delete_points)
    monkeypatch.setattr("app.core.s3.delete_prefix", must_not_delete_prefix)


async def test_delete_is_logical_and_fast(client, monkeypatch):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-1", "org-1", "user-1"),
        _chunk("chunk-1", "doc-1", "org-1", "user-1", version=1, start_offset=0, end_offset=5),
        _chunk("chunk-2", "doc-1", "org-1", "user-1", version=1, start_offset=6, end_offset=11),
    )
    invalidate_calls: list = []
    _forbid_route_physical_cleanup(monkeypatch)

    async def fake_invalidate(org_id, doc_id):
        invalidate_calls.append((org_id, doc_id))

    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", fake_invalidate)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.delete("/documents/doc-1", headers=_auth(token))

    assert resp.status_code == 200
    assert resp.json() == {"doc_id": "doc-1", "status": "deleted"}
    assert invalidate_calls == [("org-1", "doc-1")]
    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-1")
        assert doc is not None
        assert doc.status == DocumentStatus.DELETING
        assert doc.failure_reason is None
        rows = (await session.execute(select(Chunk).where(Chunk.doc_id == "doc-1"))).scalars().all()
        assert rows == []


async def test_delete_cache_failure_responds_success_with_note(client, monkeypatch):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-1", "org-1", "user-1"),
        _chunk("chunk-1", "doc-1", "org-1", "user-1", version=1, start_offset=0, end_offset=5),
    )
    _forbid_route_physical_cleanup(monkeypatch)

    async def failing_invalidate(org_id, doc_id):
        raise RuntimeError("redis down")

    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", failing_invalidate)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.delete("/documents/doc-1", headers=_auth(token))

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "doc-1"
    assert body["status"] == "deleted"
    assert body["note"] == "changes may take a few minutes to fully take effect"
    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-1")
        assert doc is not None
        assert doc.status == DocumentStatus.DELETING
        assert doc.failure_reason is None
        rows = (await session.execute(select(Chunk).where(Chunk.doc_id == "doc-1"))).scalars().all()
        assert rows == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/documents/doc-a"),
        ("get", "/documents/doc-a/chunks"),
        ("delete", "/documents/doc-a"),
    ],
)
async def test_model_a_strict_owner_scoping(client, monkeypatch, method, path):
    await _seed(
        _org("org-1"),
        _user("user-a"),
        _user("user-b"),
        _doc("doc-a", "org-1", "user-a"),
        _chunk("chunk-1", "doc-a", "org-1", "user-a", version=1, start_offset=0, end_offset=5),
    )
    _forbid_route_physical_cleanup(monkeypatch)

    async def fake_invalidate(org_id, doc_id):
        pass

    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", fake_invalidate)

    colleague = create_access_token(sub="user-b", org_id="org-1")
    resp = await getattr(client, method)(path, headers=_auth(colleague))

    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"

    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-a")
        assert doc is not None
        assert doc.status == DocumentStatus.EMBEDDED
        rows = (await session.execute(select(Chunk).where(Chunk.doc_id == "doc-a"))).scalars().all()
        assert [c.id for c in rows] == ["chunk-1"]

    owner = create_access_token(sub="user-a", org_id="org-1")
    resp = await getattr(client, method)(path, headers=_auth(owner))
    assert resp.status_code == 200


async def test_delete_already_deleting_doc_is_idempotent(client, monkeypatch):
    await _seed(
        _org("org-1"),
        _user("user-1"),
        _doc("doc-1", "org-1", "user-1", DocumentStatus.DELETING),
        _chunk("chunk-1", "doc-1", "org-1", "user-1", version=1, start_offset=0, end_offset=5),
    )
    invalidate_calls: list = []
    _forbid_route_physical_cleanup(monkeypatch)

    async def fake_invalidate(org_id, doc_id):
        invalidate_calls.append((org_id, doc_id))

    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", fake_invalidate)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.delete("/documents/doc-1", headers=_auth(token))

    assert resp.status_code == 200
    assert resp.json() == {"doc_id": "doc-1", "status": "deleted"}
    assert invalidate_calls == [("org-1", "doc-1")]
    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-1")
        assert doc is not None
        assert doc.status == DocumentStatus.DELETING
        rows = (await session.execute(select(Chunk).where(Chunk.doc_id == "doc-1"))).scalars().all()
        assert rows == []


async def test_delete_document_other_org_hidden_and_untouched(client, monkeypatch):
    await _seed(_org("org-2"), _user("user-2"), _doc("doc-2", "org-2", "user-2"))
    _forbid_route_physical_cleanup(monkeypatch)

    async def must_not_invalidate(org_id, doc_id):
        raise AssertionError("invalidate_for_doc must not be called")

    monkeypatch.setattr("app.api.routes_documents.invalidate_for_doc", must_not_invalidate)

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await client.delete("/documents/doc-2", headers=_auth(token))

    assert resp.status_code == 422
    assert resp.json()["error"] == "document not found"
    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-2")
        assert doc is not None
        assert doc.status == DocumentStatus.EMBEDDED


async def test_cancel_document_never_resurrects_deleting(client, monkeypatch):
    doc = _doc("doc-1", "org-1", "user-1", DocumentStatus.DELETING)
    doc.pending_version = 2
    await _seed(
        _org("org-1"),
        _user("user-1"),
        doc,
        _chunk("chunk-v1", "doc-1", "org-1", "user-1", version=1, start_offset=0, end_offset=5),
        _chunk("chunk-v2", "doc-1", "org-1", "user-1", version=2, start_offset=0, end_offset=5),
    )

    async def must_not_publish(payload, routing_key=None):
        raise AssertionError("cancel must not publish for a DELETING document")

    def must_not_delete_points(point_ids, payload_filter=None):
        raise AssertionError("cancel must not delete points for a DELETING document")

    monkeypatch.setattr(cancel_module, "publish_message", must_not_publish)
    monkeypatch.setattr(cancel_module, "delete_points", must_not_delete_points)

    await cancel_module.cancel_document("doc-1")

    async with db_module.get_session() as session:
        doc = await session.get(Document, "doc-1")
        assert doc is not None
        assert doc.status == DocumentStatus.DELETING
        assert doc.pending_version == 2
        rows = (
            (await session.execute(select(Chunk.id).where(Chunk.doc_id == "doc-1"))).scalars().all()
        )
        assert sorted(rows) == ["chunk-v1", "chunk-v2"]


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
