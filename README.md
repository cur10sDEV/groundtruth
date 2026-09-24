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

# 4. frontend (node 18+)
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
| `WEBHOOK_SECRET` | optional | shared secret for the MinIO webhook (see below) |

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
| ingestion worker | – | RabbitMQ consumer + stale-version cleanup loop |
| frontend (Next.js) | 3000 | chat UI, document upload |
| postgres | 5432 | documents, chunks, memberships, query logs (dbs: rag, flagsmith, flag_engine, langfuse — auto-created on first boot) |
| minio | 9000 / 9001 | document storage (S3 API / console) |
| qdrant | 6333 | hybrid vector + sparse index |
| redis | 6379 | semantic cache, rate limiting |
| rabbitmq | 5672 / 15672 | ingestion queue, retry + DLQ (mgmt UI) |
| flagsmith / edge | 8000 / 8001 | feature flags |
| langfuse | 3002 | LLM tracing UI (via compose overlay) |
| prometheus / grafana | 9090 / 3001 | metrics scraping + dashboards |

Upload flow: `POST /documents/upload` stores the object in MinIO and publishes
an ingest message to RabbitMQ (direct publish is the canonical path); the
worker parses, chunks, embeds and upserts to Qdrant + Postgres, flipping the
document version on success. Re-uploading a file with the same name creates a
new version of that document; old versions keep serving until the flip, and a
periodic job cleans stale chunks/points afterwards.

The MinIO webhook (`POST /internal/minio-event`) is an **optional** alternative
ingest trigger: it is authenticated (`X-Webhook-Secret` header must match
`WEBHOOK_SECRET`; an unset secret disables the endpoint) and is not wired into
the MinIO container by default.
