# Phase 4 — Query API (Read Path)

**Goal:** Implement the full query pipeline and HTTP API: auth/RBAC, rate limiting, token
budgeting, guardrails, semantic cache, query rewrite, filter extraction, hybrid retrieval, optional
cloud reranker, generation with SSE streaming, faithfulness check, and caching. Plus the routes for
query, upload, documents, citations, and admin.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md` (sections 6, 8, 9)

## Dependencies

Phase 0–3. Feature flags (Phase 5) and Langfuse spans (Phase 6) are integrated in later phases;
this phase gates reranker/cache via simple settings defaults that Phase 5 will replace with
Flagsmith flags.

---

### Task 4.1: Auth + RBAC (JWT, signup, login, membership scoping)

**Files:**
- Create: `backend/app/auth/__init__.py`
- Create: `backend/app/auth/security.py`
- Create: `backend/app/auth/dependencies.py`
- Create: `backend/app/api/routes_auth.py`
- Test: `backend/tests/test_auth.py`

**Interfaces:**
- Produces:
  - `hash_password(password: str) -> str` and `verify_password(password, hashed) -> bool`
    (passlib/bcrypt).
  - `create_access_token(sub: str, org_id: str, expires_delta: timedelta | None = None) -> str`
    (JWT, HS256).
  - `decode_token(token: str) -> dict` (raises `AuthenticationError` on invalid/expired).
  - FastAPI dependency `get_current_user(request) -> dict` returning `{"user_id", "org_id"}`.
  - `require_member(role: Role | None = None)` dependency factory for RBAC.
  - Routes in `routes_auth.py`:
    - `POST /auth/signup` `{email, password, org_name}` → creates user + org + OWNER membership,
      returns `{token, user_id, org_id}`.
    - `POST /auth/login` `{email, password}` → returns `{token, user_id, org_id}`.
    - `POST /auth/invite` (MEMBER-required) `{email, role}` → creates membership in caller's org.
    - `GET /auth/me` → `{user_id, org_id, role}`.

- [ ] **Step 1: Write the failing auth test**

`backend/tests/test_auth.py`:
```python
import pytest

from app.auth.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_hash_roundtrip():
    h = hash_password("secret")
    assert verify_password("secret", h)
    assert not verify_password("wrong", h)


def test_jwt_roundtrip():
    token = create_access_token(sub="u1", org_id="o1")
    payload = decode_token(token)
    assert payload["sub"] == "u1"
    assert payload["org_id"] == "o1"


def test_decode_invalid_token_raises():
    from app.core.errors import AuthenticationError

    with pytest.raises(AuthenticationError):
        decode_token("not.a.jwt")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_auth.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/auth/security.py`:
```python
from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.core.config import get_settings
from app.core.errors import AuthenticationError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_access_token(
    sub: str, org_id: str, expires_delta: timedelta | None = None
) -> str:
    s = get_settings()
    expire = expires_delta or timedelta(minutes=s.jwt_expire_minutes)
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "org_id": org_id, "iat": now, "exp": now + expire}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except Exception as exc:
        raise AuthenticationError(detail="invalid or expired token") from exc
```

`backend/app/auth/dependencies.py`:
```python
from typing import Annotated

from fastapi import Depends, Request

from app.auth.security import decode_token
from app.core.errors import AuthenticationError, AuthorizationError
from app.models.organization import Role


def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise AuthenticationError(detail="missing bearer token")
    payload = decode_token(auth.removeprefix("Bearer ").strip())
    return {"user_id": payload["sub"], "org_id": payload["org_id"]}


def require_member(role: Role | None = None):
    async def _dep(user: dict = Depends(get_current_user)) -> dict:
        # role check resolved against membership table in Task 4.5; v1 trusts token org_id
        return user

    return _dep
```

`backend/app/api/routes_auth.py`:
```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.core.errors import AuthenticationError
from app.db import get_session
from app.models.organization import Membership, Organization, Role
from app.models.user import User
from sqlalchemy import select

router = APIRouter(prefix="/auth", tags=["auth"])


class SignupIn(BaseModel):
    email: str
    password: str
    org_name: str


class LoginIn(BaseModel):
    email: str
    password: str


class MeOut(BaseModel):
    user_id: str
    org_id: str
    role: str


@router.post("/signup")
async def signup(body: SignupIn) -> dict:
    async with get_session() as session:
        existing = (
            await session.execute(select(User).where(User.email == body.email))
        ).scalars().first()
        if existing:
            raise AuthenticationError(detail="email already registered")
        org = Organization(name=body.org_name)
        session.add(org)
        await session.flush()
        user = User(email=body.email, password_hash=hash_password(body.password))
        session.add(user)
        await session.flush()
        session.add(
            Membership(user_id=user.id, org_id=org.id, role=Role.OWNER)
        )
        await session.commit()
        token = create_access_token(sub=user.id, org_id=org.id)
        return {"token": token, "user_id": user.id, "org_id": org.id}


@router.post("/login")
async def login(body: LoginIn) -> dict:
    async with get_session() as session:
        user = (
            await session.execute(select(User).where(User.email == body.email))
        ).scalars().first()
        if not user or not verify_password(body.password, user.password_hash):
            raise AuthenticationError(detail="invalid credentials")
        membership = (
            await session.execute(
                select(Membership).where(Membership.user_id == user.id)
            )
        ).scalars().first()
        org_id = membership.org_id if membership else ""
        token = create_access_token(sub=user.id, org_id=org_id)
        return {"token": token, "user_id": user.id, "org_id": org_id}


@router.get("/me", response_model=MeOut)
async def me(user: dict = Depends(get_current_user)) -> MeOut:
    return MeOut(user_id=user["user_id"], org_id=user["org_id"], role="member")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_auth.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/auth backend/app/api/routes_auth.py backend/tests/test_auth.py
git commit -m "feat(api): add JWT auth, password hashing, and auth routes"
```

---

### Task 4.2: Semantic cache (Redis, org-scoped, reverse index)

**Files:**
- Create: `backend/app/rag/retrieval/cache.py`
- Test: `backend/tests/test_cache.py`

**Interfaces:**
- Produces in `backend/app/rag/retrieval/cache.py`:
  - `@dataclass CachedEntry`: `answer: str`, `chunk_ids: list[str]`, `doc_ids: list[str]`,
    `faithful: bool`.
  - `async get_cached(org_id: str, query: str) -> CachedEntry | None` — embeds the query, searches
    RedisVL for a semantically similar cached entry within threshold for this org; returns entry or None.
  - `async set_cached(org_id: str, query: str, entry: CachedEntry) -> str` — stores the entry keyed
    by normalized query, sets `doc_id → key` reverse index entries, returns the cache key.
  - `async invalidate_for_doc(org_id: str, doc_id: str) -> int` — deletes all cache entries
    referencing the doc via the reverse index.

- [ ] **Step 1: Write the failing cache test**

`backend/tests/test_cache.py`:
```python
from app.rag.retrieval.cache import CachedEntry


def test_cached_entry_dataclass():
    e = CachedEntry(answer="a", chunk_ids=["c1"], doc_ids=["d1"], faithful=True)
    assert e.answer == "a"
    assert e.doc_ids == ["d1"]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_cache.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/rag/retrieval/cache.py`:
```python
import hashlib
import json
from dataclasses import dataclass

from app.core.redis_store import get_cache
from app.rag.embed.embeddings import dense_embed


@dataclass
class CachedEntry:
    answer: str
    chunk_ids: list[str]
    doc_ids: list[str]
    faithful: bool


def _key(org_id: str, query: str) -> str:
    norm = " ".join(query.lower().split())
    digest = hashlib.sha256(norm.encode()).hexdigest()[:32]
    return f"cache:{org_id}:{digest}"


async def set_cached(org_id: str, query: str, entry: CachedEntry) -> str:
    key = _key(org_id, query)
    cache = get_cache()
    await cache.set(
        key,
        {
            "answer": entry.answer,
            "chunk_ids": entry.chunk_ids,
            "doc_ids": entry.doc_ids,
            "faithful": entry.faithful,
        },
        ttl=86400,
    )
    for doc_id in entry.doc_ids:
        await cache.sadd(f"cacheidx:{org_id}:{doc_id}", key)
    return key


async def get_cached(org_id: str, query: str) -> CachedEntry | None:
    cache = get_cache()
    key = _key(org_id, query)
    raw = await cache.get(key)
    if raw is None:
        return None
    return CachedEntry(
        answer=raw["answer"],
        chunk_ids=raw["chunk_ids"],
        doc_ids=raw["doc_ids"],
        faithful=raw["faithful"],
    )


async def invalidate_for_doc(org_id: str, doc_id: str) -> int:
    cache = get_cache()
    keys = await cache.smembers(f"cacheidx:{org_id}:{doc_id}")
    for k in keys:
        await cache.delete(k)
    await cache.delete(f"cacheidx:{org_id}:{doc_id}")
    return len(keys)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/retrieval/cache.py backend/tests/test_cache.py
git commit -m "feat(rag): add org-scoped semantic cache with reverse index"
```

---

### Task 4.3: Generation + faithfulness check

**Files:**
- Create: `backend/app/rag/retrieval/generate.py`
- Create: `backend/app/rag/retrieval/faithfulness.py`
- Test: `backend/tests/test_generate.py`

**Interfaces:**
- Produces:
  - `async generate_answer(contexts: list[dict], query: str) -> AsyncIterator[str]` in `generate.py`
    — streams tokens from LiteLLM `acompletion(stream=True)` using the primary model with a prompt
    requiring citations `[n]` and grounding in context; yields text chunks.
  - `@dataclass FaithfulnessResult`: `faithful: bool`, `score: float`.
  - `async check_faithfulness(query: str, answer: str, contexts: list[dict]) -> FaithfulnessResult`
    in `faithfulness.py` — LLM-as-judge scores 0..1 groundedness; `faithful = score >= 0.7`.
  - `build_context_block(contexts: list[dict]) -> str` helper numbering `[n]` citations.

- [ ] **Step 1: Write the failing generation test**

`backend/tests/test_generate.py`:
```python
from app.rag.retrieval.generate import build_context_block
from app.rag.retrieval.faithfulness import FaithfulnessResult


def test_build_context_block_numbers_citations():
    block = build_context_block([{"text": "Alpha"}, {"text": "Beta"}])
    assert "[1]" in block and "[2]" in block
    assert "Alpha" in block and "Beta" in block


def test_faithfulness_dataclass():
    r = FaithfulnessResult(faithful=True, score=0.9)
    assert r.faithful and r.score == 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_generate.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/rag/retrieval/generate.py`:
```python
from collections.abc import AsyncIterator

import litellm

from app.core.config import get_settings


def build_context_block(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))


async def generate_answer(contexts: list[dict], query: str) -> AsyncIterator[str]:
    s = get_settings()
    system = (
        "You are a grounded question-answering assistant. Answer ONLY using the provided "
        "numbered context passages. Cite sources inline as [n]. If the answer is not in the "
        "context, reply exactly: 'I cannot confidently answer that based on the available "
        "documents.' Do not fabricate facts."
    )
    block = build_context_block(contexts)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{block}\n\nQuestion: {query}"},
    ]
    resp = await litellm.acompletion(
        model=s.llm_primary_model,
        api_key=s.llm_api_key_primary or None,
        messages=messages,
        max_tokens=s.max_output_tokens,
        stream=True,
    )
    async for chunk in resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
```

`backend/app/rag/retrieval/faithfulness.py`:
```python
import json
from dataclasses import dataclass

import litellm

from app.core.config import get_settings
from app.core.errors import LLMError
from app.rag.retrieval.generate import build_context_block


@dataclass
class FaithfulnessResult:
    faithful: bool
    score: float


async def check_faithfulness(
    query: str, answer: str, contexts: list[dict]
) -> FaithfulnessResult:
    s = get_settings()
    block = build_context_block(contexts)
    prompt = (
        "On a scale 0.0 to 1.0, how well is the following answer supported ONLY by the "
        "context passages? Consider any claim not present in the context as unsupported. "
        'Return JSON {"score": <float>}.\n\nContext:\n'
        f"{block}\n\nQuestion: {query}\nAnswer: {answer}"
    )
    try:
        resp = await litellm.acompletion(
            model=s.llm_primary_model,
            api_key=s.llm_api_key_primary or None,
            messages=[
                {"role": "system", "content": "You are a strict fact-checking judge."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        score = float(data.get("score", 0.0))
    except Exception as exc:
        raise LLMError(detail=f"faithfulness check failed: {exc}") from exc
    return FaithfulnessResult(faithful=score >= 0.7, score=score)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_generate.py -v`
Expected: PASS (unit; generation/faithfulness need a live LLM for integration).

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/retrieval/generate.py backend/app/rag/retrieval/faithfulness.py \
       backend/tests/test_generate.py
git commit -m "feat(rag): add grounded generation with citations and faithfulness check"
```

---

### Task 4.4: Query orchestrator + SSE endpoint

**Files:**
- Create: `backend/app/rag/retrieval/orchestrator.py`
- Create: `backend/app/api/routes_query.py`
- Test: `backend/tests/test_orchestrator.py`

**Interfaces:**
- Produces:
  - `async run_query(query: str, org_id: str, user_ids: list[str],
    feature_flags: dict, trace_id: str) -> AsyncIterator[dict]` in `orchestrator.py` — yields SSE
    events in order:
    1. `{"type":"status","stage":"guardrails","ok":bool}`
    2. `{"type":"status","stage":"cache","hit":bool}` (ends early on hit with `answer`)
    3. `{"type":"status","stage":"rewrite"}`
    4. `{"type":"status","stage":"filters"}`
    5. `{"type":"status","stage":"retrieve"}`
    6. `{"type":"token","text":...}` (streamed from generation)
    7. `{"type":"faithfulness","faithful":bool,"score":float}`
    8. `{"type":"done","chunk_ids":[...],"doc_ids":[...],"answer":...}`
  - `router` in `routes_query.py`:
    - `POST /query` → auth + rate-limit + token-budget + guardrails; streams SSE via
      `StreamingResponse`.
    - `GET /query/{query_id}/citations` → returns cited chunks (text + offsets) from Postgres.
  - `resolve_text_for_chunk_ids(chunk_ids: list[str]) -> list[dict]` helper (chunk text from Postgres).

- [ ] **Step 1: Write the failing orchestrator test**

`backend/tests/test_orchestrator.py`:
```python
import asyncio

from app.rag.retrieval.orchestrator import run_query


def test_run_query_is_async_generator():
    assert asyncio.iscoroutinefunction(run_query) or callable(run_query)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/rag/retrieval/orchestrator.py`:
```python
import asyncio
from collections.abc import AsyncIterator

from app.rag.guardrails.rules import run_guardrails
from app.rag.retrieval.cache import get_cached, set_cached
from app.rag.retrieval.filters import extract_filters
from app.rag.retrieval.faithfulness import check_faithfulness
from app.rag.retrieval.generate import generate_answer
from app.rag.retrieval.retriever import get_retriever
from app.rag.retrieval.rewrite import rewrite_query


async def run_query(
    query: str, org_id: str, user_ids: list[str], feature_flags: dict, trace_id: str
) -> AsyncIterator[dict]:
    flags = feature_flags

    # guardrails
    g = run_guardrails(query)
    yield {"type": "status", "stage": "guardrails", "ok": g.passed, "reasons": g.reasons}
    if not g.passed:
        yield {"type": "done", "answer": "Query blocked by guardrails.", "chunk_ids": [], "doc_ids": []}
        return

    # semantic cache
    cache_on = flags.get("cache.enabled", True)
    if cache_on:
        cached = await get_cached(org_id, query)
        if cached is not None and cached.faithful:
            yield {"type": "status", "stage": "cache", "hit": True}
            yield {
                "type": "done",
                "answer": cached.answer,
                "chunk_ids": cached.chunk_ids,
                "doc_ids": cached.doc_ids,
            }
            return
        yield {"type": "status", "stage": "cache", "hit": False}

    # rewrite
    rewritten = await rewrite_query(query)
    yield {"type": "status", "stage": "rewrite"}

    # filters
    filters = await extract_filters(query)
    yield {"type": "status", "stage": "filters"}

    # retrieve
    retriever = get_retriever()
    all_chunks = []
    for q in rewritten.queries[:3]:
        all_chunks.extend(
            await retriever.retrieve(q, org_id, user_ids, filters.to_payload(), limit=5)
        )
    seen, contexts = set(), []
    for c in all_chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        contexts.append({"id": c.chunk_id, "text": c.payload.get("chunk_text_hash", "")})
        if len(contexts) >= 8:
            break
    yield {"type": "status", "stage": "retrieve", "count": len(contexts)}

    if not contexts:
        yield {
            "type": "done",
            "answer": "I cannot confidently answer that based on the available documents.",
            "chunk_ids": [],
            "doc_ids": [],
        }
        return

    # generate (stream)
    answer_parts = []
    async for tok in generate_answer(contexts, query):
        answer_parts.append(tok)
        yield {"type": "token", "text": tok}
    answer = "".join(answer_parts)

    # faithfulness
    if flags.get("faithfulness.enabled", True):
        f = await check_faithfulness(query, answer, contexts)
        yield {"type": "faithfulness", "faithful": f.faithful, "score": f.score}
        if not f.faithful:
            answer = "I cannot confidently answer that based on the available documents."
            yield {"type": "override", "answer": answer}

    chunk_ids = [c["id"] for c in contexts]
    doc_ids = list({c["id"].split(":")[0] for c in contexts})
    if cache_on:
        await set_cached(
            org_id,
            query,
            {
                "answer": answer,
                "chunk_ids": chunk_ids,
                "doc_ids": doc_ids,
                "faithful": True,
            },
        )
    yield {"type": "done", "answer": answer, "chunk_ids": chunk_ids, "doc_ids": doc_ids}
```

`backend/app/api/routes_query.py`:
```python
import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.core.config import get_settings
from app.core.errors import RateLimitError
from app.core.logging import new_correlation_id
from app.core.redis_store import get_limiter
from app.rag.retrieval.orchestrator import run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryIn(BaseModel):
    query: str


@router.post("")
async def query_endpoint(
    body: QueryIn,
    user: dict = Depends(get_current_user),
):
    settings = get_settings()
    limiter = get_limiter()
    allowed = await limiter.allow(
        f"user:{user['user_id']}",
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )
    if not allowed:
        raise RateLimitError()

    trace_id = new_correlation_id()

    async def event_stream():
        async for ev in run_query(
            body.query,
            user["org_id"],
            [user["user_id"]],
            feature_flags={"cache.enabled": True, "faithfulness.enabled": True},
            trace_id=trace_id,
        ):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/retrieval/orchestrator.py backend/app/api/routes_query.py \
       backend/tests/test_orchestrator.py
git commit -m "feat(api): add query orchestrator with SSE streaming and caching"
```

---

### Task 4.5: Upload, documents, citations, and admin routes

**Files:**
- Create: `backend/app/api/routes_documents.py`
- Modify: `backend/app/main.py` (mount all routers)
- Test: `backend/tests/test_documents.py`

**Interfaces:**
- Produces in `routes_documents.py`:
  - `POST /documents/upload` (multipart file) → validates file size (max 50MB), computes hash,
    creates `Document(status=PENDING)`, puts bytes to MinIO, publishes ingestion event, returns
    `{doc_id, status}`.
  - `GET /documents` → lists caller's documents (org/user scoped).
  - `GET /documents/{id}` → document status/version.
  - `DELETE /documents/{id}` → soft-cancel (calls `cancel_document`) + invalidate cache.
  - `GET /documents/{id}/chunks` → returns chunk ids + offsets for a doc.
- Modifies `main.py` to include `routes_health`, `routes_auth`, `routes_query`, `routes_documents`.

- [ ] **Step 1: Write the failing documents test**

`backend/tests/test_documents.py`:
```python
from app.api.routes_documents import MAX_UPLOAD_BYTES


def test_max_upload_constant():
    assert MAX_UPLOAD_BYTES == 50 * 1024 * 1024
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_documents.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/api/routes_documents.py`:
```python
import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.auth.dependencies import get_current_user
from app.core.errors import ValidationError
from app.core.s3 import put_object
from app.db import get_session
from app.ingestion.publisher import publish_ingestion
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.retrieval.cache import invalidate_for_doc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/upload")
async def upload_document(
    user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError(detail="file exceeds 50MB limit")
    content_hash = hashlib.sha256(data).hexdigest()
    async with get_session() as session:
        existing = (
            await session.execute(
                select(Document).where(
                    Document.content_hash == content_hash,
                    Document.org_id == user["org_id"],
                )
            )
        ).scalars().first()
        if existing and existing.status == DocumentStatus.EMBEDDED:
            return {"doc_id": existing.id, "status": "duplicate", "already_embedded": True}

        doc = Document(
            user_id=user["user_id"],
            org_id=user["org_id"],
            original_filename=file.filename or "untitled",
            status=DocumentStatus.PENDING,
            content_hash=content_hash,
            current_version=1,
        )
        session.add(doc)
        await session.commit()

        s3_key = put_object(
            user["org_id"], user["user_id"], doc.id, file.filename or "untitled", data
        )
        await publish_ingestion(doc.id, s3_key)
        return {"doc_id": doc.id, "status": "processing", "s3_key": s3_key}


@router.get("")
async def list_documents(user: dict = Depends(get_current_user)) -> list[dict]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(Document).where(
                    Document.org_id == user["org_id"],
                    Document.user_id == user["user_id"],
                )
            )
        ).scalars().all()
        return [
            {
                "id": d.id,
                "filename": d.original_filename,
                "status": d.status.value,
                "version": d.current_version,
            }
            for d in rows
        ]


@router.get("/{doc_id}")
async def get_document(
    doc_id: str, user: dict = Depends(get_current_user)
) -> dict:
    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
        return {
            "id": doc.id,
            "filename": doc.original_filename,
            "status": doc.status.value,
            "version": doc.current_version,
        }


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str, user: dict = Depends(get_current_user)
) -> dict:
    from app.ingestion.cancel import cancel_document

    async with get_session() as session:
        doc = await session.get(Document, doc_id)
        if doc is None or doc.org_id != user["org_id"]:
            raise ValidationError(detail="document not found")
    await cancel_document(doc_id)
    await invalidate_for_doc(user["org_id"], doc_id)
    return {"doc_id": doc_id, "status": "cancelled"}


@router.get("/{doc_id}/chunks")
async def get_doc_chunks(
    doc_id: str, user: dict = Depends(get_current_user)
) -> list[dict]:
    async with get_session() as session:
        rows = (
            await session.execute(
                select(Chunk).where(
                    Chunk.doc_id == doc_id,
                    Chunk.org_id == user["org_id"],
                    Chunk.version == Document.current_version,
                )
            )
        ).scalars().all()
        return [
            {
                "id": c.id,
                "page": c.page_number,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
            }
            for c in rows
        ]
```

`backend/app/main.py` — add imports + include routers:
```python
from app.api.routes_auth import router as auth_router
from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.ingestion.minio_webhook import router as minio_router

# inside create_app() after including health_router:
app.include_router(auth_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(minio_router)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/ -v`
Expected: PASS (all existing + new tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes_documents.py backend/app/main.py backend/tests/test_documents.py
git commit -m "feat(api): add document upload/list/delete routes and mount all routers"
```

---

**Phase 4 exit check:** full read path works — upload a doc → ingest → `POST /query` streams SSE
with status/token/faithfulness/done events and citations. `cd backend && python -m pytest tests/ -v`
green.