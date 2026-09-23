import json

import pytest
from httpx import ASGITransport, AsyncClient

import app.db as db_module
from app.auth.security import create_access_token
from app.db import init_db
from app.main import create_app
from app.models.chunk import Chunk
from app.models.citation import Citation
from app.models.document import Document, DocumentStatus
from app.models.organization import Organization
from app.models.user import User


class _StubLimiter:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls: list[tuple] = []

    async def allow(self, user_key, limit, window_seconds):
        self.calls.append((user_key, limit, window_seconds))
        return self.allowed


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/routes_query.db")
    limiter = _StubLimiter()
    monkeypatch.setattr("app.api.routes_query.get_limiter", lambda: limiter)
    db_module._engine = None
    db_module._sessionmaker = None
    await init_db()
    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, limiter
    if db_module._engine is not None:
        await db_module._engine.dispose()
    db_module._engine = None
    db_module._sessionmaker = None


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def test_post_query_streams_sse_events(client, monkeypatch):
    c, limiter = client
    captured = {}

    async def stub_run_query(query, org_id, user_ids, feature_flags, trace_id):
        captured.update(
            query=query,
            org_id=org_id,
            user_ids=user_ids,
            feature_flags=feature_flags,
            trace_id=trace_id,
        )
        yield {"type": "status", "stage": "guardrails", "ok": True}
        yield {"type": "done", "answer": "hi", "chunk_ids": [], "doc_ids": []}

    monkeypatch.setattr("app.api.routes_query.run_query", stub_run_query)
    token = create_access_token(sub="user-1", org_id="org-1")

    resp = await c.post("/query", json={"query": "what is rbac?"}, headers=_auth(token))

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in resp.text.split("\n\n") if ln]
    assert lines == [
        'data: {"type": "status", "stage": "guardrails", "ok": true}',
        'data: {"type": "done", "answer": "hi", "chunk_ids": [], "doc_ids": []}',
    ]
    for ln in lines:
        assert ln.startswith("data: ")
        json.loads(ln.removeprefix("data: "))
    assert captured["query"] == "what is rbac?"
    assert captured["org_id"] == "org-1"
    assert captured["user_ids"] == ["user-1"]
    assert captured["feature_flags"] == {"cache.enabled": True, "faithfulness.enabled": True}
    assert captured["trace_id"]  # fresh correlation id per request
    # rate limit is keyed to the user and checked before streaming
    assert limiter.calls == [("user:user-1", 30, 60)]


async def test_post_query_requires_auth(client):
    c, _ = client
    resp = await c.post("/query", json={"query": "what is rbac?"})
    assert resp.status_code == 401


async def test_post_query_rate_limited_returns_429(client, monkeypatch):
    c, _ = client
    blocked = _StubLimiter(allowed=False)
    monkeypatch.setattr("app.api.routes_query.get_limiter", lambda: blocked)
    called = []

    async def stub_run_query(*args, **kwargs):
        called.append((args, kwargs))
        yield {"type": "done"}

    monkeypatch.setattr("app.api.routes_query.run_query", stub_run_query)
    token = create_access_token(sub="user-1", org_id="org-1")

    resp = await c.post("/query", json={"query": "what is rbac?"}, headers=_auth(token))

    assert resp.status_code == 429
    assert resp.json()["error"] == "rate limit exceeded"
    assert called == []


async def test_post_query_rejects_over_budget_query(client, monkeypatch):
    c, _ = client
    called = []

    async def stub_run_query(*args, **kwargs):
        called.append((args, kwargs))
        yield {"type": "done"}

    monkeypatch.setattr("app.api.routes_query.run_query", stub_run_query)
    token = create_access_token(sub="user-1", org_id="org-1")

    resp = await c.post("/query", json={"query": "x" * 40000}, headers=_auth(token))

    assert resp.status_code == 422
    assert resp.json()["error"] == "query exceeds input token budget"
    assert called == []


async def test_citations_returns_org_scoped_chunks_with_offsets(client):
    c, _ = client
    async with db_module.get_session() as session:
        session.add(Organization(id="org-1", name="A"))
        session.add(Organization(id="org-2", name="B"))
        session.add(User(id="user-1", email="cite@example.com", password_hash="x"))
        session.add(
            Document(
                id="doc-1",
                user_id="user-1",
                org_id="org-1",
                original_filename="a.txt",
                status=DocumentStatus.EMBEDDED,
                content_hash="0" * 64,
            )
        )
        session.add(
            Document(
                id="doc-2",
                user_id="user-1",
                org_id="org-2",
                original_filename="b.txt",
                status=DocumentStatus.EMBEDDED,
                content_hash="1" * 64,
            )
        )
        session.add(
            Chunk(
                id="chunk-1",
                doc_id="doc-1",
                user_id="user-1",
                org_id="org-1",
                chunk_text="hello world",
                start_offset=0,
                end_offset=11,
            )
        )
        session.add(
            Chunk(
                id="chunk-2",
                doc_id="doc-2",
                user_id="user-1",
                org_id="org-2",
                chunk_text="other org text",
                start_offset=0,
                end_offset=14,
            )
        )
        session.add(Citation(id="cit-1", query_id="q-1", chunk_id="chunk-1", doc_id="doc-1"))
        session.add(Citation(id="cit-2", query_id="q-1", chunk_id="chunk-2", doc_id="doc-2"))
        await session.commit()

    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await c.get("/query/q-1/citations", headers=_auth(token))

    assert resp.status_code == 200
    # chunk-2 belongs to org-2: invisible even though cited under the same query_id
    assert resp.json() == [
        {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "text": "hello world",
            "start_offset": 0,
            "end_offset": 11,
        }
    ]


async def test_citations_unknown_query_returns_empty_list(client):
    c, _ = client
    token = create_access_token(sub="user-1", org_id="org-1")
    resp = await c.get("/query/never-seen/citations", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == []


async def test_citations_requires_auth(client):
    c, _ = client
    resp = await c.get("/query/q-1/citations")
    assert resp.status_code == 401
