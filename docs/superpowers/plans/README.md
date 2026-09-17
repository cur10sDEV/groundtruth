# RAG Prod — Implementation Plans (Phase-wise)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement these plans task-by-task. Tasks use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, end-to-end RAG system (ingestion → hybrid retrieval →
generation → validation) with full backend, Next.js frontend, and observability, runnable via
docker-compose as a local reference.

**Architecture:** Two Python runtimes (FastAPI API server for the read path, an ingestion worker
for the write path) sharing a `rag` package, wired to Postgres, Qdrant, MinIO, Redis, and
RabbitMQ, with Langfuse + Prometheus/Grafana for observability and self-hosted Flagsmith for
feature flags. Hand-rolled RAG pipeline for control over grounding/citations.

**Tech Stack:** Python 3.12 / FastAPI / Pydantic v2, SQLAlchemy, Qdrant, MinIO, Redis, RabbitMQ,
LiteLLM (primary + fallback cloud models), fastembed (client-side BM25 sparse), Flagsmith,
Langfuse, Prometheus/Grafana, Next.js / React / Tailwind / React Query / SSE.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md`

## Plan index

Each phase produces independently testable software. Build in order; each phase depends only on
the ones before it.

| Phase | File | Deliverable |
|---|---|---|
| 0 | `phase-0-infra-scaffolding.md` | Repo config, settings/logging, full docker-compose, health checks |
| 1 | `phase-1-storage-layer.md` | SQLAlchemy models+migrations, MinIO, Qdrant, Redis clients |
| 2 | `phase-2-core-rag-package.md` | Chunkers, embeddings, guardrails, retriever, filter extraction, query rewrite |
| 3 | `phase-3-ingestion-worker.md` | Parsers, ingestion pipeline, RabbitMQ consumer, versioning, cleanup, cancellation |
| 4 | `phase-4-query-api.md` | Auth/RBAC, rate limiting, semantic cache, generation+faithfulness, SSE, API routes |
| 5 | `phase-5-feature-flags.md` | Flagsmith integration (backend + frontend) |
| 6 | `phase-6-observability.md` | Langfuse spans, Prometheus metrics, structured logging, Grafana dashboards |
| 7 | `phase-7-frontend.md` | Next.js app: auth, chat+SSE, upload, citations |
| 8 | `phase-8-eval-redteam.md` | RAG eval harness + seed QA set, red-team script |

## Global Constraints

Applies to every phase/task (copied verbatim from the spec):

- **Zero hallucination / 100% correctness**: never emit a claim not traceable to a retrieved
  chunk; on failure cascade an explicit "cannot confidently answer" message, never a guess.
- **Information isolation**: RBAC-scoped retrieval, cache, storage. No cross-user/org leaks.
- **Versioned documents**: reads use `documents.current_version`; new versions write alongside and
  atomically flip; stale data removed by a periodic cleanup job.
- **Ingestion latency**: a just-updated document is searchable within ~15 minutes.
- **LLM routing**: one primary cloud model + one fallback cloud model via LiteLLM; no local model
  runtime. LLM timeouts/rate limits → retries + exponential backoff → fallback model → explicit error.
- **Reranker**: cloud (e.g., Cohere Rerank), gated by `reranker.enabled` feature flag, disabled by default.
- **Sparse vectors**: generated client-side with fastembed `Qdrant/bm25` (self-hosted Qdrant has no
  server-side text→sparse inference); Qdrant native `SparseVectorParams(modifier=IDF)`.
- **Metadata filters**: extracted from the query by an LLM tool call into a structured `Filters`
  object, applied to BOTH dense and sparse search scoped within RBAC bounds.
- **Step-level observability**: every pipeline stage is a Langfuse span (input/output/duration);
  typed domain errors; global handler returns safe JSON with trace ID.
- **Feature flags via Flagsmith (self-hosted)**: backend local-evaluation (~5s refresh), frontend
  via `flagsmith-js-client`/`@flagsmith/react`; `default_flag_handler` fallback so flag-service
  outage never hard-fails. Flags: `reranker.enabled`, `cache.enabled`, `multi_query.enabled`,
  `filter_extraction.enabled`, `faithfulness.enabled`.
- **Secrets**: never hard-code API keys; all via env (pydantic-settings), `.env.example` committed.
- **Tooling**: ruff for lint, pytest for tests, structured JSON logs with correlation IDs.
- **S3 key scheme**: `s3://<bucket>/<org_id>/<user_id>/<doc_id>/<uuid>.ext`; original filename in
  metadata only; versioning on; lifecycle deletes non-current versions after 30 days.
- **Formatting/naming**: Python type hints everywhere; `ruff format` compliant code.