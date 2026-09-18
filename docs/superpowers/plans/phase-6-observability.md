# Phase 6 — Observability

**Goal:** Add step-level Langfuse traces to every pipeline stage, Prometheus metrics, structured
JSON logging with correlation IDs, and Grafana dashboards, plus a `/metrics` endpoint.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md` (section 7, step-level observability)

## Dependencies

Phase 0–5. Langfuse (`rag-langfuse` :3000), Prometheus (`rag-prometheus` :9090), Grafana
(`rag-grafana` :3001) running.

---

### Task 6.1: Langfuse trace helper + span decorator

**Files:**
- Create: `backend/app/core/telemetry.py`
- Test: `backend/tests/test_telemetry.py`

**Interfaces:**
- Produces in `backend/app/core/telemetry.py`:
  - `get_langfuse() -> Langfuse` singleton (guarded: if keys unset, returns a `None`-safe no-op
    wrapper so the app runs without Langfuse configured).
  - `@dataclass SpanContext` / context manager `trace_step(name: str, trace_id: str,
    parent: SpanContext | None = None)` yielding a span with `update(input=..., output=...)` and
    auto-recording exceptions as error spans.
  - `ensure_trace(trace_id: str) -> SpanContext` to create the root trace for a request.

- [ ] **Step 1: Write the failing telemetry test**

`backend/tests/test_telemetry.py`:
```python
from app.core.telemetry import get_langfuse, trace_step


def test_get_langfuse_returns_client():
    # no-op safe when keys unset
    assert get_langfuse() is not None


def test_trace_step_is_context_manager():
    cm = trace_step(name="test", trace_id="t1")
    assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_telemetry.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/core/telemetry.py`:
```python
import logging
from contextlib import contextmanager
from typing import Any, Iterator

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class SpanContext:
    def __init__(self, name: str, trace_id: str) -> None:
        self.name = name
        self.trace_id = trace_id
        self._langfuse = None
        self._generation = None

    def update(self, **kwargs: Any) -> None:
        if self._generation is not None:
            self._generation.update(**kwargs)

    def end(self, **kwargs: Any) -> None:
        if self._generation is not None:
            self._generation.end(**kwargs)


class _NoopSpan(SpanContext):
    def update(self, **kwargs: Any) -> None:  # type: ignore[override]
        return

    def end(self, **kwargs: Any) -> None:
        return


def get_langfuse():
    s = get_settings()
    if not s.langfuse_public_key or not s.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("langfuse init failed", extra={"exc": str(exc)})
        return None


def ensure_trace(trace_id: str) -> SpanContext:
    lf = get_langfuse()
    if lf is None:
        return _NoopSpan("trace", trace_id)
    generation = lf.generation(name=f"query:{trace_id}", trace_id=trace_id)
    ctx = SpanContext(f"query:{trace_id}", trace_id)
    ctx._langfuse = lf
    ctx._generation = generation
    return ctx


@contextmanager
def trace_step(name: str, trace_id: str, parent: SpanContext | None = None) -> Iterator[SpanContext]:
    lf = get_langfuse()
    if lf is None:
        yield _NoopSpan(name, trace_id)
        return
    generation = lf.generation(
        name=name,
        trace_id=trace_id,
        parent_observation_id=getattr(parent, "_generation", None).id if parent and parent._generation else None,
    )
    ctx = SpanContext(name, trace_id)
    ctx._langfuse = lf
    ctx._generation = generation
    try:
        yield ctx
        generation.end(output={"ok": True})
    except Exception as exc:
        generation.end(metadata={"error": str(exc)}, level="ERROR")
        raise
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_telemetry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/telemetry.py backend/tests/test_telemetry.py
git commit -m "feat(obs): add Langfuse trace helper with no-op fallback"
```

---

### Task 6.2: Instrument the query orchestrator with spans

**Files:**
- Modify: `backend/app/rag/retrieval/orchestrator.py`
- Test: `backend/tests/test_orchestrator.py` (extend)

**Interfaces:**
- Produces: each stage in `run_query` is wrapped in `with trace_step("stage_name", trace_id):`
  recording input/output. Stages: `guardrails`, `cache`, `rewrite`, `filters`, `retrieve`,
  `rerank`, `generate`, `faithfulness`.

- [ ] **Step 1: Write a span-presence test**

Extend `backend/tests/test_orchestrator.py`:
```python
from app.core.telemetry import trace_step


def test_trace_step_records_ok_without_langfuse():
    with trace_step("stage", "trace-1") as span:
        span.update(output={"n": 1})
    assert span.name == "stage"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_orchestrator.py -v`
Expected: FAIL (trace_step not yet imported/available in test env).

- [ ] **Step 3: Wrap stages with trace_step**

In `orchestrator.py`, add at top:
```python
from app.core.telemetry import trace_step
```
Wrap the guardrail, cache, rewrite, filters, retrieve, rerank, generate, and faithfulness blocks
with `with trace_step("guardrails", trace_id):` etc. (Task 6.1 gives the API.)

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/rag/retrieval/orchestrator.py backend/tests/test_orchestrator.py
git commit -m "feat(obs): wrap query pipeline stages in Langfuse spans"
```

---

### Task 6.3: Prometheus metrics + /metrics endpoint

**Files:**
- Create: `backend/app/core/metrics.py`
- Create: `backend/app/api/routes_metrics.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_metrics.py`

**Interfaces:**
- Produces in `metrics.py`:
  - `REQUESTS_TOTAL = Counter(...)`
  - `ERRORS_TOTAL = Counter(...)`
  - `LATENCY = Histogram(...)`
  - `TOKENS_IN`, `TOKENS_OUT` Counters
  - `CACHE_HITS`, `CACHE_MISSES` Counters
  - `INGESTION_PROCESSED`, `INGESTION_FAILED` Counters
  - `record_request(duration_seconds, status)` helper; `incr_cache(hit: bool)`; `record_tokens(in, out)`.
- `routes_metrics.py`: `GET /metrics` returning `generate_latest()` with `prometheus_client` content type.
- Modifies `main.py`: include metrics router.

- [ ] **Step 1: Write the failing metrics test**

`backend/tests/test_metrics.py`:
```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_metrics_endpoint():
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_requests_total" in resp.text
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/core/metrics.py`:
```python
from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter("rag_requests_total", "Total requests")
ERRORS_TOTAL = Counter("rag_errors_total", "Total errors")
LATENCY = Histogram("rag_latency_seconds", "Request latency", buckets=[0.1, 0.25, 0.5, 1, 2, 5])
TOKENS_IN = Counter("rag_tokens_in_total", "Input tokens")
TOKENS_OUT = Counter("rag_tokens_out_total", "Output tokens")
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Cache misses")
INGESTION_PROCESSED = Counter("rag_ingestion_processed_total", "Docs ingested")
INGESTION_FAILED = Counter("rag_ingestion_failed_total", "Docs failed")


def record_request(duration_seconds: float, status: str) -> None:
    REQUESTS_TOTAL.inc()
    LATENCY.observe(duration_seconds)
    if status != "ok":
        ERRORS_TOTAL.inc()


def incr_cache(hit: bool) -> None:
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()


def record_tokens(tokens_in: int, tokens_out: int) -> None:
    TOKENS_IN.inc(tokens_in)
    TOKENS_OUT.inc(tokens_out)
```

`backend/app/api/routes_metrics.py`:
```python
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Update `backend/app/main.py`:
```python
from app.api.routes_metrics import router as metrics_router
app.include_router(metrics_router)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/metrics.py backend/app/api/routes_metrics.py backend/app/main.py \
       backend/tests/test_metrics.py
git commit -m "feat(obs): add Prometheus metrics and /metrics endpoint"
```

---

### Task 6.4: Grafana dashboard

**Files:**
- Create: `infra/grafana/dashboards/rag.json`

**Interfaces:**
- Produces a Grafana dashboard JSON (provisioned via the datasource in Phase 0) with panels for:
  request totals/rate, error rate, latency p50/p95/p99 (via `rag_latency_seconds` histogram), token
  usage, cache hit/miss rate, and ingestion throughput.

- [ ] **Step 1: Write the dashboard JSON**

`infra/grafana/dashboards/rag.json` — a minimal valid Grafana dashboard referencing the Prometheus
datasource and the metrics from Task 6.3. Example panel:
```json
{
  "title": "RAG Prod",
  "uid": "rag-prod",
  "panels": [
    {
      "title": "Requests/s",
      "type": "timeseries",
      "datasource": { "type": "prometheus", "uid": "" },
      "targets": [{ "expr": "rate(rag_requests_total[1m])" }],
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 }
    },
    {
      "title": "Error rate",
      "type": "timeseries",
      "datasource": { "type": "prometheus", "uid": "" },
      "targets": [{ "expr": "rate(rag_errors_total[1m]) / rate(rag_requests_total[1m])" }],
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 }
    }
  ],
  "templating": { "list": [] },
  "annotations": { "list": [] },
  "schemaVersion": 39
}
```

- [ ] **Step 2: Validate JSON**

Run: `python -m json.tool infra/grafana/dashboards/rag.json > /dev/null && echo valid`
Expected: `valid`.

- [ ] **Step 3: Commit**

```bash
git add infra/grafana/dashboards/rag.json
git commit -m "feat(obs): add Grafana dashboard for RAG metrics"
```

---

**Phase 6 exit check:** query pipeline produces nested Langfuse spans; `/metrics` returns
Prometheus data; Grafana at `:3001` shows the dashboard; `cd backend && python -m pytest tests/ -v`
green.

---

### Task 6.5: Enhanced /health readiness checks

**Files:**
- Modify: `backend/app/api/routes_health.py`
- Test: `backend/tests/test_health.py`

**Interfaces:**
- Produces: `/health` returns `{"status": "ok"|"degraded", "checks": {database, qdrant, redis,
  flagsmith, langfuse}}` — each check probes the component and reports readiness without failing
  the request (all `@pytest.mark.integration` or gracefully handled).

- [ ] **Step 1: Write the failing readiness test**

Append to `backend/tests/test_health.py`:
```python
def test_health_includes_checks():
    from fastapi.testclient import TestClient
    from app.main import create_app

    resp = TestClient(create_app()).get("/health")
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["checks"]) >= {"database", "qdrant", "redis"}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_health.py -v`
Expected: FAIL — `checks` key missing.

- [ ] **Step 3: Write implementation**

Replace `backend/app/api/routes_health.py` with a readiness probe:
```python
from fastapi import APIRouter

from app.core.config import get_settings
from app.core.redis_store import get_cache
from app.db import _get_engine

router = APIRouter(tags=["health"])


async def _readiness() -> dict:
    checks = {"database": True, "qdrant": True, "redis": True,
              "flagsmith": True, "langfuse": True}
    try:
        async with _get_engine().connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    except Exception:
        checks["database"] = False
    try:
        get_cache()
    except Exception:
        checks["redis"] = False
    # Qdrant / Flagsmith / Langfuse probes added in Phase 1/5/6 once clients exist;
    # each wraps its client call in try/except and sets the flag.
    return checks


@router.get("/health")
async def health() -> dict:
    checks = await _readiness()
    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/test_health.py -v`
Expected: PASS (health returns 200 with a `checks` map; `degraded` when local services are down).

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes_health.py backend/tests/test_health.py
git commit -m "feat(obs): add component readiness checks to /health"
```