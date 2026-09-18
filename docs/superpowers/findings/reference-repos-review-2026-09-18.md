# Reference Repo Review — LLM-Layer Security Findings

**Date:** 2026-09-18
**Reviewer:** opencode (for review by human before applying to the RAG prod plan)
**Status:** ✅ **Reviewed and approved by user; findings applied 2026-09-18.**
**Repos reviewed:**
1. `d-hackmt/8hr-MARATHON` — production Enterprise Agentic RAG API (FastAPI + LangGraph + NeMo Guardrails + Qdrant + Portkey gateway)
2. `pdichone/lang-production-api` — production chat API (FastAPI + LangGraph + custom security pipeline + caching + slowapi)

**Scope:** Lower-level implementation details, focused on LLM-layer security: guardrails, PII
detection/masking, prompt-injection, output validation, plus adjacent production details (model
fallback, caching, metrics, evals).

> This is a findings/recommendation document for your review. **No plan or spec files have been
> changed.** After you review, we decide which findings to fold into the plan.

---

## Summary of what was reviewed

**8hr-MARATHON** key files:
- `app/guardrails/rails.py`, `app/guardrails/colang_rules.py` — NeMo Guardrails gate
- `app/main.py` — guardrails as Gate 1, then LangGraph
- `app/gateway/client.py` — Portkey LLM gateway (fallback + retry + cache)
- `app/agents/graph.py`, `nodes/responder.py` — LangGraph orchestration, context truncation
- `app/services/retrieval/ranking_service.py` — FlashRank local cross-encoder reranker
- `app/ingestion/processor.py`, `chunking/splitter.py` — ingestion
- `evals/guardrails_eval.py`, `evals/metrics.py` — TP/TN/FP/FN guardrail eval, RAG metrics

**lang-production-api** key files:
- `app/security.py` — `InputSanitizer`, `PIIDetector`, `OutputValidator`, `SecurityPipeline`
- `app/main.py` — wiring: security → cache → agent → output validation → cache store
- `app/agent.py` — LangGraph retry/fallback/error nodes
- `app/cache.py`, `app/monitoring.py`, `app/config.py`, `app/models.py`
- `tests/test_security.py` — deterministic guardrail tests (no LLM calls)

---

## Findings

### 1. PII: detect AND mask, on BOTH input and output (HIGH VALUE)

`lang-production-api` does not merely *detect* PII to block it — it **masks** it, in two places:

- **Input masking:** before the user query reaches the LLM, PII is replaced with redaction markers,
  e.g. `john@test.com` → `[EMAIL REDACTED]`. This means the LLM never even sees the raw PII.
- **Output masking:** after generation and **before returning to the client**, the response is
  scanned again and any PII (e.g. the LLM echoing a PII-bearing chunk retrieved from the index) is
  masked in place.

`PIIDetector.PATTERNS` + `MASK_MAP` (`security.py:56-90`): email, phone, SSN, **credit card**.
`OutputValidator.validate` (`security.py:96-131`) runs PII detection + masking on output and appends
warnings.

**Implication for our plan:** Our Phase 2 guardrails only *detect* PII and reject the whole query.
That is blunt — a legitimate query containing an email gets blocked entirely, and a RAG answer that
echoes a PII chunk is returned raw. We should add **masking** (not just blocking) on input, and an
**output PII mask** step before streaming the final answer, including **credit card** in the
patterns. This directly serves our "no data leak" guarantee. **Recommend adding this.**

### 2. Output security validation (HIGH VALUE — a real gap in our plan)

`langprod` validates the **generated output** for:
- PII leakage (masked, above)
- Harmful content patterns (`security.py:102-106`): `here's how to hack/steal/attack`,
  `password is `, `api[_\s]?key[=:]` — i.e. it guards against the LLM **exfiltrating secrets or
  emitting harmful instructions** in its answer.

Our plan has a *faithfulness* check (grounding) but **no output security validation** for
PII/secret/harmful leakage. These are complementary:
- Faithfulness = is it grounded in retrieved context?
- Output validation = is the answer leaking PII/secrets/harmful content?

**Implication:** Add an `output_validate` stage (mask PII, scan for secret/harmful patterns) as the
final checkpoint before returning/streaming the answer. **Recommend adding.**

### 3. Input sanitization beyond detection: delimiter/template cleaning (MEDIUM-HIGH)

`InputSanitizer.clean()` (`security.py:45-50`):
- strips runs of `---` / `===` (the classic "END OF PROMPT" / prompt-boundary injection),
- escapes `{{` `}}` to `{ {` `} }` (prevents template-injection into the prompt/structured-output
  formatting).

Our plan currently only *detects* injection phrases; it does not **neutralize** boundary/format
markers. Worth adding a `clean()` pass so a benign query that happens to contain prompt-boundary
syntax can be sanitized rather than rejected.

**Implication:** Add input `clean()` (delimiter/template neutralization) alongside detection.
**Recommend adding.**

### 4. Dedicated guard model vs. heuristic regex (MEDIUM — design decision)

The two repos take opposite approaches, and both are worth weighing against our Phase 2 regex guardrails:

- **langprod = pure regex heuristics** (fast, free, deterministic, fully testable offline). Good
  for high-confidence patterns, zero latency/cost, easy unit tests.
- **8hr-MARATHON = NeMo Guardrails** with a **dedicated, cheap guard LLM**
  (`llama-3.1-8b-instant`, `temperature=0`) running intent classification / jailbreak / off-topic
  detection (`rails.py:20-24`, `colang_rules.py`). More robust against novel phrasing, but adds an
  LLM call (latency + cost) and needs careful eval.

Our plan currently uses regex only. **Recommendation:** keep the deterministic regex layer as a fast
first gate (cheap, testable), and optionally add a **cheap dedicated guard model** behind it —
natural fit for our LiteLLM routing (a low-cost primary/fallback cloud model dedicated to
classification, `temperature=0`). I'd make this a **feature flag** (`guard_model.enabled`) per our
Flagsmith setup.

### 5. Detecting whether a guardrail "fired" — rail indicators (MEDIUM)

With a model-based guardrail you can't just inspect the model's output structure; you need a signal.
`8hr-MARATHON` defines `RAIL_INDICATORS` (`colang_rules.py:119-125`): distinctive substrings from
each refusal message, and checks `any(indicator in content for indicator in RAIL_INDICATORS)` to
conclude a rail fired (`rails.py:56`). This is a clean, robust pattern for gating on model output.

**Implication:** If we add a model-based guard, define explicit refusal markers and gate on those,
rather than trusting the guard model's JSON shape. **Adopt this pattern.**

### 6. Guardrails eval as binary classification (MEDIUM — improves our red-team)

`8hr-MARATHON/evals/guardrails_eval.py` runs a labeled dataset against the **live** `/query` endpoint
and classifies each as **TP / TN / FP / FN**, then computes **precision, recall, accuracy**. It
checks whether the response's `thought_process` contains `"guardrails fired"` — i.e. it asserts the
*gate actually fired*, not just that the output looked OK.

Our plan's red-team script only asserts "was blocked"; it doesn't measure **false positives**
(legitimate queries wrongly blocked) or **precision/recall** of the guardrail itself. That's
important because over-aggressive guardrails silently hurt usability.

**Implication:** Add a proper guardrail eval (labeled positives + negatives → precision/recall) to
Phase 8. **Recommend adding.**

### 7. Two-stage model fallback tracked in state (MEDIUM — refinement)

`langprod` disables SDK retries (`max_retries=0`) and implements fallback in the **agent graph**
(`agent.py`): `process → fallback → error` nodes, tracking `retry_count` and `model_used`, with an
explicit `handle_error` node returning a graceful message. It surfaces `model_used` and `error` in
the response.

Our plan relies on LiteLLM's built-in fallback. Marathon also uses Portkey's gateway-level fallback.
**Implication:** whichever we use, surface `model_used` in the response/trace and ensure a graceful
fallback-error message — worth making explicit in the spec's error-handling section (already close).

### 8. Context truncation to a hard budget (MEDIUM)

`8hr-MARATHON` `responder.py:35-43` truncates the assembled context to `max_context_chars = 25000`
to respect Groq TPM limits, and logs when truncated. This is a concrete guard against the number of
retrieved chunks blowing past the model's token window.

**Implication:** Our plan sets `max_output_tokens` but doesn't hard-cap assembled **context** tokens.
Add a context token budget that truncates/enforces before generation. **Recommend adding.**

### 9. Health endpoint reports component readiness (LOW-MEDIUM)

`langprod` `/health` returns `checks: {agent, security, cache}` with `status: healthy|degraded`
(`main.py:221-238`). Our plan's `/health` only returns `{"status":"ok"}`.

**Implication:** Enhance `/health` to report readiness of core components (DB, Qdrant, Redis,
Flagsmith, Langfuse). Cheap, useful for the docker-compose health checks and debugging. **Recommend adding.**

### 10. Rate limiting + standardized 429 handling (LOW — we already cover)

`langprod` uses `slowapi` `@limiter.limit` + a global `RateLimitExceeded` handler returning
`{"error", "detail"}`. We use Redis sliding-window already; the only takeaway is the consistent 429
JSON shape (we have this via `RateLimitError`). No change needed.

### 11. Reranker: FlashRank local vs. our cloud Cohere (LOW — informational)

`8hr-MARATHON` uses **FlashRank** (local ONNX cross-encoder, `ms-marco-MiniLM-L-6-v2`) with a
**fallback to original Qdrant order** on any failure (`ranking_service.py:64-67`). You chose a cloud
reranker (Cohere) — that's fine. The transferable pattern is **fail-open to retrieval order** when
the reranker errors (we already sketched this in Phase 5; keep it).

### 12. Payload/metadata hygiene (LOW — we already do it better)

`8hr-MARATHON` stores the **full chunk text in the Qdrant payload** (`processor.py:83-86`). Our plan
already stores only vectors + metadata in Qdrant and keeps chunk text in Postgres — which is the
better practice. No change; noting for completeness.

---

## Recommended priority for folding into the plan

| # | Finding | Priority | Proposed action |
|---|---|---|---|
| 1 | Input PII masking (not just blocking) | High | Add masking pass in Phase 2 guardrails |
| 2 | Output security validation (PII/secret/harmful) | High | Add `output_validate` stage in Phase 4 orchestrator |
| 3 | Input cleaning (delimiter/template neutralization) | Med-High | Add `clean()` alongside detection |
| 4 | Dedicated guard model behind regex | Med | Optional, feature-flagged; use rail-indicator gate |
| 5 | Rail-indicator gating pattern | Med | Adopt if model-based guard added |
| 6 | Guardrail precision/recall eval (TP/TN/FP/FN) | Med | Extend Phase 8 eval |
| 7 | Surface `model_used` + graceful fallback message | Med | Refine error-handling section |
| 8 | Hard context-token budget before generation | Med | Add context truncation to Phase 4 |
| 9 | Enhanced `/health` readiness checks | Low-Med | Extend Phase 0 `/health` |
| 10 | Reranker fail-open | Low | Already planned; keep |
| 11 | Context/metadata hygiene | Low | Already planned; keep |
| 12 | Standardized 429 JSON | Low | Already planned; keep |

---

## Not applying (your existing choices are fine)
- **Local FlashRank reranker** — you chose cloud Cohere (kept).
- **NeMo Guardrails / Colang** — heavier dependency; recommend the leaner custom + optional model
  guard instead, but this is your call to weigh.
- **Portkey gateway** — we use LiteLLM routing; equivalent capability, no change.

---

**Next step:** Review the table above and tell me which findings to apply. I'll then update the
spec (`docs/superpowers/specs/2026-09-18-rag-prod-design.md`) and the relevant phase files
(`phase-2-core-rag-package.md`, `phase-4-query-api.md`, `phase-8-eval-redteam.md`, and/or
`phase-0-infra-scaffolding.md`) accordingly — nothing is changed until you approve.

---

## Applied changes (2026-09-18)

All 12 findings approved. The following were folded into the plan (the context limit is
configurable via `max_context_tokens`, default 4000):

| # | Finding | Where applied |
|---|---|---|
| 1 | Input PII masking (not just blocking) | Phase 2 Task 2.3 — `mask_pii`, `run_guardrails` masks PII |
| 2 | Output security validation | Phase 2 Task 2.3 `validate_output`; Phase 4 orchestrator step 10 |
| 3 | Input cleaning (delimiter/template neutralization) | Phase 2 Task 2.3 `clean_input` |
| 4 | Dedicated guard model | Phase 5 Task 5.4 `model_guard`, gated by `guard_model.enabled` flag + `GUARD_MODEL` config |
| 5 | Rail-indicator gating pattern | Phase 5 Task 5.4 `RAIL_INDICATORS` |
| 6 | Guardrail precision/recall eval | Phase 8 Task 8.4 `guardrails_eval.py` (TP/TN/FP/FN) |
| 7 | Surface `model_used` | Phase 4 Tasks 4.3–4.4 (`generate_answer` meta event + `model_used` in SSE) |
| 8 | Hard context-token budget | Phase 4 Task 4.3 `truncate_contexts` + config `max_context_tokens` (configurable) |
| 9 | Enhanced `/health` readiness | Phase 6 Task 6.5 (`checks` map) |
| 10 | Reranker fail-open | Already planned; kept |
| 11 | Context/metadata hygiene | Already planned; kept |
| 12 | Standardized 429 JSON | Already planned; kept |