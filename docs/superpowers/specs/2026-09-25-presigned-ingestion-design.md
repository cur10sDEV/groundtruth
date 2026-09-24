# Presigned Ingestion Pipeline — Design Spec

**Date:** 2026-09-25
**Status:** Approved design (user-approved sections: eager signing, FAILED-on-duplicate user-scoped, Option 1 translator, true deletion with guarded flip, webhook removal)
**Supersedes:** the multipart upload route and the MinIO webhook as ingestion triggers
**Related:** `2026-09-18-rag-prod-design.md` (base system spec)

## 1. Goals

1. The API server never buffers document bytes: browsers upload directly to MinIO via **presigned POST** with a server-enforced size cap.
2. Ingestion triggers via **MinIO → RabbitMQ direct AMQP notification** (no HTTP webhook, no app in the trigger path).
3. **No noise enters the work stream**: MinIO-side event narrowing (`--event put --prefix documents/`) plus a translator that validates every event before anything reaches the `ingestion` queue or its retry/DLQ topology.
4. **True document deletion**: linearize with one atomic transaction, finish the cascade idempotently; deletion can never be outraced by an in-flight ingestion.
5. Duplicate content (same bytes, same user, different doc) is marked `FAILED` with a visible reason; the duplicate check becomes **(org, user)-scoped** (fixing the latent org-scoped hand-back bug).
6. The MinIO webhook is removed from everywhere.

## 2. Non-goals

- Direct MinIO-object manipulation (console/CLI) triggering app behavior — storage is not the lifecycle owner; app routes own document lifecycle. Documented as a contract, not wired.
- Cross-document dedup for *same-doc re-versions*: a new version of the same doc re-ingests even if bytes are identical (unchanged behavior).
- Multi-worker concurrency: single worker process; at-least-once delivery is made harmless by idempotency (deterministic chunk IDs already landed).

## 3. Upload flow — presigned POST (Option A: eager row)

`POST /documents/upload` becomes a signing endpoint (same path, changed semantics):

1. Auth + RBAC as today (`get_current_user`; org scoping).
2. **Versioning decision (unchanged logic, no bytes needed):** if an `EMBEDDED` doc exists with the same `(org_id, user_id, original_filename)`, set that doc `status=PENDING`, `pending_version = current_version + 1` (commit) and sign for IT; else create a fresh `Document` row (`status=PENDING`, `current_version=1`) and sign for it.
3. Generate the S3 key as today (`build_key`: `documents/{org}/{user}/{doc_id}/{uuid}.{ext}`).
4. `generate_presigned_post(bucket, key, Conditions=[["content-length-range", 0, <upload_max_bytes>]], ExpiresIn=<presign_expiry_seconds>)`.
5. Respond `200 {"doc_id", "status", "upload": {"url", "fields"}}`. **No file bytes pass through the API. No message is published** — the event stream is the only trigger.

**Presigned POST (not PUT) is deliberate:** POST policies are the only S3/MinIO mechanism where the server enforces `content-length-range`; a presigned PUT cannot enforce the size cap.

The browser then `FormData`-POSTs (fields + file) directly to MinIO.

## 4. Trigger — MinIO → RabbitMQ direct

`infra/minio-init/init-bucket.sh` (extended, idempotent, after the existing bucket setup):

```sh
mc admin config set local notify_amqp:primary \
  url="amqp://${RABBIT_USER}:${RABBIT_PASS}@rabbitmq:5672" \
  exchange="minio.events" exchange_type="direct" \
  routing_key="minio.events" durable="on" \
  queue_dir="/tmp/minio-events" queue_limit="10000"
mc admin service restart local
mc event add local/documents arn:minio:sqs::primary:amqp \
  --event put --prefix documents/ --ignore-existing
```

- `--event put --prefix documents/`: only `ObjectCreated` events under our key space are ever emitted. Deletes/gets never reach RabbitMQ.
- `queue_dir` + `queue_limit`: MinIO persists undelivered events to disk — events survive RabbitMQ downtime.
- `durable="on"`: the `minio.events` exchange is durable; the translator declares/binds the queue idempotently at startup.
- Compose: the `minio-init` service gains `depends_on: rabbitmq (service_healthy)`; RabbitMQ credentials injected via env (defaults guest/guest for the local reference).

## 5. Translator — `app/ingestion/event_translator.py`

A second asyncio task in the existing worker process (`worker.main`: `gather(consume_loop(), translate_loop(), cleanup_job_loop())`). Consumes the durable `minio.events` queue and is the **only** publisher into `ingestion`.

Per event, in order:
1. Parse JSON; require an `s3:ObjectCreated:*` eventName (belt-and-braces) and the `documents` bucket.
2. Parse the key with `key_to_parts`; malformed → **drop**.
3. Load the `Document` row: missing → **drop**; `status` not in `{PENDING, PROCESSING}` (e.g. `DELETING`, `EMBEDDED`, `FAILED`) → **drop**.
4. **Ownership check:** key's `(org_id, user_id)` must equal the row's → mismatch → **drop**.
5. Publish `{"doc_id", "s3_key", "new_version": <row.pending_version or None>}` to `ingestion` (existing `publish_ingestion`), then ack.

Every drop: warning log + `rag_events_dropped_total` counter (new metric, `reason` label). Drops are terminal (no retry/DLQ for noise — that is the point of the layer).

## 6. Consumer changes

`process_message` skip-guard set gains `DELETING` (skip + ack alongside the existing EMBEDDED/FAILED handling; the versioned-recovery exception for FAILED stands — versioned messages still proceed for FAILED rows, but never for DELETING).

## 7. Duplicate content — FAILED with visible reason

In `pipeline._ingest`, after fetch + hash, **before** `_reset_version` and any embedding work:

1. Query for another `EMBEDDED` doc with the same `content_hash` in the same **`(org_id, user_id)`** (excluding self).
2. If found: mark this row `FAILED` with `failure_reason = f"duplicate of {existing_id}"` (new column — see §10), best-effort `delete_object(s3_key)` (the just-uploaded blob), and stop — no chunking, no embedding, no reset. Ack (the doc stays in the list showing the reason; the existing version, if any, is untouched because this ran before `_reset_version`).
3. Same-doc, same-bytes, new version: unchanged (re-ingests fully — §2 non-goal).

## 8. Reaper — cleanup job extensions

The existing `cleanup_job_loop` gains two idempotent sweeps:

1. **Abandoned uploads:** rows with `status=PENDING` older than `reaper_pending_after_seconds` (default 3600; presigned URLs expire in 900s) → prefix-clean any blobs (`delete_prefix`), then delete the row. Covers "event dropped" corners: late events find no row and are dropped by the translator.
2. **Deletion tombstones:** rows with `status=DELETING` → run the deletion cascade remainder (§9), then delete the row.

## 9. True deletion — linearize, then reap

`DELETE /documents/{id}`:

**Step 1 — one atomic transaction (the linearization point, milliseconds):**
```sql
BEGIN;
  UPDATE documents SET status='DELETING' WHERE id=? AND org_id=?;
  DELETE FROM chunks WHERE doc_id=?;   -- ALL versions
COMMIT;
```
Instantly: not listed (list filters `DELETING`), not retrievable (resolve needs chunk rows), and nothing new can go live (guarded flip below).

**Step 2 — best-effort, same request, idempotent, safe order:**
Qdrant points by `doc_id` payload filter (all versions) → cache entries (`invalidate_for_doc`) → S3 blobs (`delete_prefix`) → delete the Document row. Any failure leaves the `DELETING` tombstone for the reaper; the route returns `200 {"doc_id", "status": "deleted"}` once Step 1 committed (logical deletion is already true).

**The guarded flip (keystone, in `pipeline`):** the version flip becomes
`UPDATE documents SET current_version=:nv, status='EMBEDDED', pending_version=NULL WHERE id=:doc AND status='PROCESSING'`
— rowcount 0 → failure path. All pipeline status writes (EMBEDDED/FAILED/PROCESSING commits) get the same guarded shape (`WHERE status='PROCESSING'` where applicable) so a `DELETING` row is never stomped. On losing the race, the pipeline's existing failure path cleans partial points; the reaper sweeps the rest. Invariant: **content only goes live at the flip, and the flip only succeeds while PROCESSING.**

`cancel_document` keeps its current contract (in-flight cancel preserves the prior version) and shares the guarded flip; nothing else changes there.

## 10. Schema, config, metrics

- **Migration 0003:** `documents.failure_reason: String | None` (nullable). Add `DELETING` to the document status enum (native-enum change if the column uses one — implementer verifies; must be idempotent like 0002).
- **Config (`app/core/config.py` + `.env.example`):** `upload_max_bytes` (move the route constant; default 50MB), `presign_expiry_seconds=900`, `reaper_pending_after_seconds=3600`.
- **Metrics:** `rag_events_dropped_total{reason}` (add to `metrics.py`; wire in the translator).
- **S3 helper:** `delete_prefix(org_id, user_id, doc_id) -> int` (list_objects_v2 by prefix + batch delete).

## 11. Frontend

- Upload button: call the signing endpoint → `FormData` POST (returned fields + file) to the returned URL → existing status polling takes over. Client-side size guard mirrors `upload_max_bytes` for fast feedback (server-enforced regardless).
- Documents list: filter `DELETING`; render `status=FAILED` docs with `failure_reason` (link to the duplicate doc when the reason parses).
- Remove the multipart-to-backend upload code path.

## 12. Webhook removal (everywhere)

Delete `app/api/.../minio_webhook.py` (endpoint + route registration + tests), `WEBHOOK_SECRET` from config/`.env.example`, the README webhook section, and stale references in plan docs.

## 13. Security recap

The presigned URL is a bearer capability for exactly one key (`documents/{org}/{user}/{doc}/{uuid}.ext`), expiring in 900s, size-capped by MinIO's own policy. Ownership is validated twice (translator + consumer, which keep existing checks). Leak impact: upload one ≤50MB file into one document slot. The webhook's unauthenticated surface is gone entirely.

## 14. Testing

**Offline unit (backend):** translator validation matrix (valid / non-created / malformed key / missing row / wrong status / ownership mismatch / forwards PENDING and PROCESSING); pipeline duplicate → `FAILED` + reason + blob deleted + no reset; reaper (PENDING>1h with blob prefix cleanup; DELETING completion); signing endpoint (auth, versioning decision, stubbed presign, key shape); guarded flip (delete-during-ingest race simulation → no resurrection, FAILED not stomped over DELETING); delete cascade order + partial-failure → tombstone → reaper completes.
**Offline unit (frontend):** tsc + build green; upload/sign flow components compile against the new response shape.
**Live integration (marked `integration`):** boot MinIO+RabbitMQ → run the real init script → verify `mc event ls` → presign via the endpoint → upload bytes → observe event → translator → `ingestion` → document EMBEDDED; then DELETE → verify points/rows/blobs gone; duplicate upload → FAILED + reason.

## 15. Migration & rollout order (for the implementation plan)

1. Pipeline primitives first (guarded flip/status writes, `failure_reason`, duplicate detection, `delete_prefix`, delete-cascade helpers) — keeps the system green throughout.
2. Signing endpoint + S3 helpers (route switches to presign; multipart path removed in the same commit as the frontend switch).
3. MinIO init wiring + translator + worker task + consumer `DELETING` guard + reaper extensions.
4. Frontend switch + `failure_reason` UI.
5. Webhook removal + README rewrite + live end-to-end verification.

Each step ships green on the full suite; the trigger handover (step 3) is the only step where the running system's ingestion behavior changes.
