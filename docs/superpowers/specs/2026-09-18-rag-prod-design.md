# RAG Prod — Production-Grade Reference Design

**Date:** 2026-09-18
**Status:** Approved for implementation planning
**Path:** Architectural (new greenfield project)

## 1. Goal & Scope

Build a production-grade, end-to-end Retrieval-Augmented Generation (RAG) system with a full
backend and frontend, observability, and a real ingestion→query pipeline. It targets a
**local, runnable reference** via docker-compose that is architecturally faithful to a
hyperscale design (10M docs, 200 QPS), without provisioning cloud managed services.

Core guarantees:
- **Zero hallucination / 100% correctness** — no claim is emitted that cannot be traced to a
  retrieved chunk; on failure we cascade an explicit "cannot confidently answer" rather than guess.
- **Information isolation** — RBAC-scoped retrieval, cache, and storage. No cross-user/org leaks.
- **Versioned documents** — reads use the `current_version`; new versions write alongside old and
  atomically flip; stale data cleaned up by a periodic job.
- **Just-updated documents** are searchable within ~15 minutes of ingestion.

### Out of scope (reference simplification)
- Real cloud provisioning (S3, SQS, managed Qdrant, etc.) — emulated locally.
- Massive horizontal scaling, multi-region, and SLA/HA guarantees.
- Billing/subscription enforcement beyond token budgeting hooks.

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Backend runtime | Python 3.12, FastAPI, Pydantic v2 |
| RAG pipeline | Hand-rolled (no LangGraph/LlamaIndex) for control over grounding/citations |
| LLM routing | LiteLLM — **one primary cloud model + one fallback cloud model** (both via API keys; no local runtime) |
| Embeddings | Cloud embedding model via LiteLLM (dense); fastembed `Qdrant/bm25` for client-side sparse (BM25) |
| Reranker | **Cloud reranker** (e.g., Cohere Rerank) behind a **feature flag** (enable/disable) |
| Vector DB | **Qdrant** — dense (cosine) + native sparse BM25 (`SparseVectorParams(modifier=IDF)`) in one collection, RRF fusion |
| Relational DB | **Postgres** (documents, chunks, users, orgs, citations, logs) |
| Object storage | **MinIO** (S3-compatible) with versioning + lifecycle |
| Cache / rate-limit | **Redis** + RedisVL (semantic cache, reverse index, sliding-window rate limit) |
| Message queue | **RabbitMQ** (durable queues + DLX for DLQ) |
| Feature flags | **Flagsmith (self-hosted)** — real-time, no-restart runtime flags with UI + audit |
| Frontend | **Next.js** (React), Tailwind, React Query, SSE streaming |
| Observability | **Langfuse** (LLM traces), OpenTelemetry + **Prometheus/Grafana** (metrics), structured JSON logs |
| Quality | ruff (lint), pytest (tests), RAG eval harness, red-team smoke script |

## 3. Repository Layout

```
rag-prod/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes (query, upload, auth, admin, health)
│   │   ├── core/           # config, logging, telemetry, errors, dependencies
│   │   ├── rag/            # shared RAG package: chunkers, retriever, guardrails, filters, eval
│   │   ├── ingestion/      # worker entrypoint + pipeline stages
│   │   ├── auth/           # JWT, RBAC, rate limiting
│   │   └── models/         # SQLAlchemy models
│   ├── tests/
│   └── pyproject.toml
├── frontend/               # Next.js app
├── infra/                  # docker-compose, dockerfiles, prometheus/grafana/langfuse/flagsmith config
├── eval/                   # QA seed set, eval harness, red-team scripts
├── user-specs/             # source research (read-only)
└── docs/superpowers/specs/
```

Two Python runtimes: **API server** (read path) and **ingestion worker** (write path), sharing
`backend/app/rag` and `backend/app/core`.

## 4. Storage & Data Model

### Feature flags & configuration
- **Flagsmith (self-hosted)** as the feature-flag system — real-time runtime toggles with no service
  restart. Runs via docker-compose (`flagsmith-api` + `flagsmith-postgres` + Flagsmith UI), with the
  optional `edge-proxy` for higher throughput.
- Flags defined once in the Flagsmith UI (with code-level defaults mirrored as `default_flag_handler`):
  `reranker.enabled`, `cache.enabled`, `multi_query.enabled`, `filter_extraction.enabled`,
  `faithfulness.enabled`, plus any stage you may later want to gate.
- **Backend (Python SDK, `flagsmith`):** local evaluation mode, `api_url` → self-hosted Flagsmith,
  short `environment_refresh_interval_seconds` (~5s) for near-instant propagation without restart.
  A `default_flag_handler` returns code-level defaults if Flagsmith is unreachable, so the app never
  hard-fails when the flag service is down.
- **Frontend (`flagsmith-js-client` / `@flagsmith/react`):** toggles UI-dependent flags live.
- Model routing (primary + fallback) and all provider API keys configured via env (pydantic-settings);
  no secrets hard-coded.

### MinIO (S3)
- Key: `s3://<bucket>/<org_id>/<user_id>/<doc_id>/<uuid>.ext`
- Original filename stored in metadata only (never as the key).
- Bucket versioning enabled as failover; lifecycle rule deletes non-current versions after 30 days.
- Uploaded files emit object-create events → RabbitMQ (ingestion trigger).

### Postgres
- `users(id, email, password_hash, created_at)`
- `organizations(id, name, created_at)`
- `memberships(user_id, org_id, role)` — RBAC; role ∈ {owner, admin, member}
- `documents(id, user_id, org_id, original_filename, status, content_hash, current_version,
  created_at, updated_at)` — status ∈ {PENDING, PROCESSING, EMBEDDED, FAILED}; `content_hash`
  dedups and short-circuits unchanged re-uploads.
- `chunks(id, doc_id, user_id, org_id, chunk_text, page_number, start_offset, end_offset,
  version, created_at, updated_at)` — denormalized user_id/org_id for filter-free scoping.
- `citations(query_id, chunk_id, doc_id)` — enables citation links on cache hits.
- `query_logs(query_id, user_id, org_id, query, cached, trace_id, created_at)`.
- `cache_index(doc_id, cache_key)` — reverse index for document-level cache invalidation.

### Qdrant
- Single collection per org namespace (or one collection with payload scoping).
- Fields per point: `id` (chunk UUID), dense vector (cosine), sparse vector (BM25), payload =
  `{doc_id, user_id, org_id, chunk_text_hash, version, page_number, chunk_index, word_count,
  char_count, type, tags, year, topic, ...}`.
- Chunk text lives in Postgres; Qdrant holds vectors + payload metadata for filtering.
- RBAC scoping enforced by payload filters on every query (org_id AND allowed user scope).

### Redis + RedisVL
- Semantic cache: key = normalized query hash; value = {answer, cited chunk_ids, cited doc_ids,
  faithfulness_result}; similarity-threshold matching; org-scoped; TTL ~1 day.
- Reverse index `doc_id → cache_keys` for immediate invalidation on document update/delete.
- Sliding-window rate limiter per user/org.

## 5. Ingestion (Write Path)

1. **Trigger:** MinIO object-create event → RabbitMQ durable queue.
2. **Dedup:** compute `content_hash`; if identical to current version, mark `EMBEDDED` and stop.
3. **Parse & clean:** PDF (PyMuPDF), DOCX (python-docx), MD/TXT (line-aware). Strip headers,
   footers, fix encoding; extract page numbers and character offsets.
4. **Structure-aware chunking:** recursive splitter respecting headings/paragraphs; target
   256–512 tokens with overlap; maintain parent→child mapping for context assembly.
5. **Metadata enrichment:** doc_id, user_id, org_id, page, offsets, chunk_index, word/char
   counts, type, tags, domain fields (year, topic).
6. **Embed:** dense via cloud model (LiteLLM); sparse via **fastembed `Qdrant/bm25`** generated client-side
   (self-hosted Qdrant does not offer server-side text→sparse inference), then upserted with Qdrant's
   native `IDF` modifier applied.
7. **Persist (version N):** insert chunks into PG and vectors into Qdrant; update
   `documents.status=EMBEDDED`.
8. **Version flip:** on full success, set `documents.current_version=N`; invalidate cache entries
   for the doc via reverse index.
9. **Cleanup job:** periodic batch deletion of chunks/vectors with `version < current_version`,
   retries + DLQ for persistent failures.
10. **Cancellation:** endpoint to abort an in-flight upload with cleanup of partial writes.

## 6. Query / Retrieval (Read Path)

```
POST /query
  1. validate (pydantic) → auth/RBAC → rate-limit (Redis) → token budget
  2. guardrails (prompt-injection, PII, profanity)          [span]
  3. semantic-cache lookup (org-scoped); serve on hit       [span]
  4. LLM query rewrite → verbose/normalized + K multi-queries [span]
  5. LLM metadata-filter extraction (tool call / structured
     output) → structured Filters object                     [span]
  6. hybrid search (dense + sparse in parallel) with
     payload filters = RBAC scope ∧ extracted filters        [span]
  7. RRF fusion → **optional cloud reranker (feature-flag-gated)**, e.g., Cohere Rerank [span]
  8. context assembly (parent chunks) → LLM generation
     (LiteLLM primary → fallback cloud model), SSE stream + sliding-window
     validation checkpoint                                   [span]
  9. faithfulness check (LLM-as-judge)                      [span]
 10. cache original query + answer + cited chunk/doc IDs +
     faithfulness result                                     [span]
 11. record trace/metrics; return answer + citations
```

### Metadata-filter extraction (step 5)
- The (rewritten) query is passed to the LLM with a tool call / structured schema describing the
  **allow-listed** filterable metadata fields (e.g., `year`, `type`, `tags`, `date_range`, and any
  registered domain fields).
- The LLM emits a structured `Filters` object, e.g. `{year: 2025, type: "report", tags: [...],
  date_range: {from, to}}`.
- Filters are applied as Qdrant payload filters **during hybrid search**, scoped within the
  user/org RBAC bounds. Only chunks whose metadata matches are considered for dense + sparse.
- If the query implies no filters, the Filters object is empty and retrieval proceeds unfiltered
  (beyond RBAC).

### Step-level observability (all steps)
- Every stage (guardrails, cache, rewrite, filter-extract, dense, sparse, RRF, rerank, generate,
  faithfulness) is a **Langfuse span** capturing input, output, duration, tokens, and model.
- A single **trace** spans the whole query; spans nest under it. Trace/span IDs flow into logs,
  metrics, and the user-facing error payload for debugging.
- Errors: each stage wraps its work in try/except → on failure records an **error span** with the
  exception, then raises a **typed domain error**; a global exception handler returns a safe JSON
  error containing the trace ID. Hallucinated/garbled output is never surfaced.

## 7. Observability

- **Langfuse:** full LLM traces (query rewrite, filter extraction, embeddings, generation,
  guardrails, faithfulness) with input/output and latency; step-by-step view for debugging.
- **Prometheus + Grafana:** request totals, error rate, latency histograms, token usage
  (input/output), cache hit/miss rate, ingestion throughput/failures, queue depth.
- **Structured JSON logs:** correlation IDs on every log line; shipped with traces.
- **DLQs:** failed ingestion jobs land in a RabbitMQ dead-letter queue for manual review.

## 8. Security / RBAC

- JWT auth; org membership + role determines retrieval and cache scope.
- RBAC payload filters on every Qdrant query; PG queries scoped by user/org.
- Semantic cache scoped by org (never served across orgs).
- Rate limiting, input/output validation, file-size limits (S3 + frontend), prompt-injection/PII/
  profanity guardrails.

## 9. Error Handling & Failure Modes

- LLM timeouts/rate limits: retries + exponential backoff, max retries, then fallback cloud model,
  then explicit error cascade.
- Storage timeouts: retry; on persistent failure return explicit error (never a guess).
- Cache down: skip cache, run full pipeline.
- Flagsmith down: SDK's `default_flag_handler` returns code-level flag defaults; pipeline continues.
- Partial write failure: events remain on the queue for reprocessing; status moved to FAILED with
  the error logged.
- Always cascade to an explicit, user-facing failure message with a trace ID — no hallucinated
  answers.

## 10. Testing & Evaluation

- **Unit:** chunkers, retriever, filter extraction, guardrails, cache, auth/RBAC.
- **Integration:** docker-compose services end-to-end (upload → ingest → query → cite).
- **Eval harness** (`eval/`): seed QA set; metrics = retrieval recall, MRR, context precision,
  answer faithfulness, answer relevance.
- **Red-team script:** prompt-injection, information-leak, bias/harmful-output smoke checks.

## 11. Defaults (set unless changed)
- Reranker: **cloud reranker (e.g., Cohere Rerank) behind a feature flag**, disabled by default;
  API key required when enabled.
- Model routing: one primary cloud model + one fallback cloud model via LiteLLM (e.g., primary
  OpenAI GPT-4o → fallback Anthropic Claude, both keyed).
- Tooling: ruff (lint), pytest (tests), monorepo layout above.
- Filter allow-list defined centrally and extendable via config.