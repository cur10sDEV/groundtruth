# rag-prod

Production-grade RAG system: FastAPI backend with streaming answers, versioned
document ingestion, guardrails, RBAC, Flagsmith feature flags, Langfuse tracing,
and a Next.js chat frontend.

## Quickstart

```bash
# 1. infrastructure (postgres, minio, qdrant, redis, rabbitmq, flagsmith, prometheus, grafana)
#    add the langfuse overlay for tracing UI (mapped on host port 3002)
docker compose -f infra/docker-compose.yml -f infra/langfuse/docker-compose.langfuse.yml up -d

# 2. backend (python 3.12)
cd backend
python3.12 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
cp .env.example .env            # then edit: set LLM_API_KEY_PRIMARY (see below)
./.venv/bin/alembic upgrade head
./.venv/bin/uvicorn app.main:create_app --factory --port 8002

# 3. ingestion worker (separate shell, same dir)
./.venv/bin/python -m app.ingestion.worker

# 4. frontend (node 20.9+)
cd ../frontend
cp .env.example .env.local
npm install
npm run dev                     # http://localhost:3000
```

## Environment setup

Copy `backend/.env.example` → `backend/.env` and `frontend/.env.example` →
`frontend/.env.local`, then fill in:

| Key | Required | Notes |
| --- | --- | --- |
| `LLM_API_KEY_PRIMARY` | yes | LiteLLM provider key for the primary model (e.g. OpenAI) |
| `LLM_API_KEY_FALLBACK` | optional | second provider; used automatically when the primary call fails |
| `RERANKER_API_KEY` | optional | Cohere key (only if `reranker.enabled` is on) |
| `FLAGSMITH_SERVER_KEY` / `NEXT_PUBLIC_FLAGSMITH_KEY` | optional | live flags; built-in defaults apply when unset |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | optional | tracing UI at http://localhost:3002 |

## Feature flags

Defaults live in `backend/app/core/flags.py` (`DEFAULT_FLAGS`) and
`frontend/lib/flags.tsx`; flip them live in the Flagsmith UI
(http://localhost:8000, server-side key) — no redeploy needed.

| Flag | Default | Controls |
| --- | --- | --- |
| `reranker.enabled` | off | Cohere reranking of retrieved contexts (fail-open) |
| `cache.enabled` | on | semantic answer cache in Redis (fail-open) |
| `multi_query.enabled` | on | multi-query retrieval expansion |
| `filter_extraction.enabled` | on | LLM-extracted metadata filters for retrieval |
| `faithfulness.enabled` | on | post-answer faithfulness judge (refusal on low score) |
| `guard_model.enabled` | off | second, model-based guardrail gate (needs `GUARD_MODEL`) |

## Eval scripts

Run from the repo root (uses the backend venv; most need live infrastructure
and a configured LLM key; `guardrails_eval` is offline):

```bash
backend/.venv/bin/python -m eval.guardrails_eval   # guardrail precision/recall (offline)
backend/.venv/bin/python -m eval.redteam           # injection / info-leak smoke tests
backend/.venv/bin/python -m eval.retrieval_eval    # retrieval quality on eval/seed_qa.json
backend/.venv/bin/python -m eval.answer_eval      # faithfulness/relevance helpers
```

## Architecture

| Service | Port | Purpose |
| --- | --- | --- |
| backend API (uvicorn) | 8002 | REST + SSE `/query`, `/documents`, `/auth`, `/health`, `/metrics` |
| ingestion worker | – | event translator + RabbitMQ consumer + cleanup/reaper loop |
| frontend (Next.js) | 3000 | chat UI, document upload |
| postgres | 5432 | documents, chunks, memberships, query logs (dbs: rag, flagsmith, flag_engine, langfuse — auto-created on first boot) |
| minio | 9000 / 9001 | document storage (S3 API / console) |
| qdrant | 6333 | hybrid vector + sparse index |
| redis | 6379 | semantic cache, rate limiting |
| rabbitmq | 5672 / 15672 | ingestion queue, retry + DLQ (mgmt UI) |
| flagsmith / edge | 8000 / 8001 | feature flags |
| langfuse | 3002 | LLM tracing UI (via compose overlay) |
| prometheus / grafana | 9090 / 3001 | metrics scraping + dashboards |

Upload: `POST /documents/sign` returns a presigned POST policy (creating a PENDING
row, or a version bump on an existing same-name document); the browser then
FormData-POSTs the file **directly to MinIO** — the backend never proxies bytes.
The 50 MB cap (`UPLOAD_MAX_BYTES`) is enforced by MinIO's POST policy itself.

Trigger: MinIO's `notify_amqp` target publishes `s3:ObjectCreated:*` events under
`documents/` to the `minio.events` exchange. The worker's translator validates
each event (event type, bucket, key shape, a matching PENDING/PROCESSING row,
key-vs-row ownership) and is the **only** publisher into the `ingestion` queue,
which the worker consumes to parse, chunk, embed and upsert to Qdrant + Postgres,
flipping the document to EMBEDDED on success. Invalid events are dropped
terminally — noise never retries, by design — and counted in the
`rag_events_dropped_total` metric on `/metrics`; a dropped event simply leaves
its row PENDING for the reaper.

Re-uploading the same filename signs a new version of that document: v1 keeps
serving until the new version flips EMBEDDED, after which a periodic job cleans
the stale chunks/points. Byte-identical content under a new name is marked
FAILED with `failure_reason: "duplicate of {id}"` and the duplicate blob is
deleted.

Deletion (`DELETE /documents/{id}`) tombstones the row as DELETING, sweeps its
chunks and best-effort invalidates cached answers, then returns immediately;
physical cleanup (Qdrant points, MinIO blobs, row removal) is owned by the
worker's reaper and completes within `CLEANUP_INTERVAL_SECONDS` (default 5 min).
If cache invalidation was unavailable, the API says so via the response's
`note` field. The same reaper resolves
rows stuck PENDING for `REAPER_PENDING_AFTER_SECONDS`: abandoned first-uploads
are deleted together with their blobs, abandoned versioned re-uploads are
restored to EMBEDDED (v1 was never touched).

Compose note: the `minio` service's `MINIO_NOTIFY_AMQP_URL_PRIMARY` interpolates
`amqp://${RABBIT_USER:-guest}:${RABBIT_PASS:-guest}@rabbitmq:5672` from the
**host** environment (your shell, or the `.env` file in the directory
`docker compose` runs from) — matching the compose `rabbitmq` service's default
guest/guest credentials. Container-level env vars never feed this interpolation;
changing broker credentials means overriding the pair on the host *and*
parameterizing the `rabbitmq` service itself.
