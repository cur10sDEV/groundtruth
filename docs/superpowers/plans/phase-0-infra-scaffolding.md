# Phase 0 — Infra Scaffolding & docker-compose

**Goal:** Lay the project skeleton, shared config/logging, and a full docker-compose that brings up
every backing service (MinIO, Postgres, Qdrant, Redis, RabbitMQ, Langfuse, Flagsmith,
Prometheus, Grafana) with health checks.

**Spec:** `docs/superpowers/specs/2026-09-18-rag-prod-design.md` (sections 2, 3, 8)

## Dependencies

Repo already initialized (`git init`, initial spec commit present). Continue from there.

---

### Task 0.1: Backend package, pyproject, and base app

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/logging.py`
- Test: `backend/tests/test_config.py`
- Test: `backend/tests/test_logging.py`
- Create: `backend/.env.example`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Settings` (pydantic-settings) class in `backend/app/core/config.py` with fields used by later
    phases (exact names locked below).
  - `get_settings()` -> `Settings` (cached via `lru_cache`).
  - `setup_logging()` in `backend/app/core/logging.py`; `get_logger(name) -> logging.Logger`.

- [ ] **Step 1: Write the failing config test**

`backend/tests/test_config.py`:
```python
from app.core.config import get_settings


def test_settings_loads_expected_fields():
    s = get_settings()
    assert s.app_name == "rag-prod"
    assert s.environment in {"development", "test", "production"}
    assert s.database_url.startswith("postgresql")


def test_settings_reads_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "override")
    assert get_settings().app_name == "override"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: app.core.config`.

- [ ] **Step 3: Write the settings and logging implementation**

`backend/pyproject.toml`:
```toml
[project]
name = "rag-prod-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "qdrant-client>=1.12",
    "boto3>=1.35",
    "redis>=5.2",
    "redisvl>=0.3",
    "aio-pika>=9.4",
    "litellm>=1.53",
    "fastembed>=0.4",
    "openai>=1.54",
    "python-multipart>=0.0.12",
    "PyJWT>=2.9",
    "passlib[bcrypt]>=1.7",
    "alembic>=1.14",
    "flagsmith>=3.8",
    "langfuse>=2.54",
    "opentelemetry-api>=1.28",
    "prometheus-client>=0.21",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "httpx>=0.27",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

`backend/app/core/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-prod"
    environment: str = "development"

    # Postgres
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"

    # MinIO (S3)
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "documents"
    s3_region: str = "us-east-1"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "chunks"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400

    # RabbitMQ
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    # LLM routing (LiteLLM). Primary + fallback.
    llm_primary_model: str = "openai/gpt-4o"
    llm_fallback_model: str = "anthropic/claude-3-5-sonnet-20241022"
    llm_api_key_primary: str = ""
    llm_api_key_fallback: str = ""
    embed_model: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-english-v3.0"
    reranker_api_key: str = ""

    # Flagsmith
    flagsmith_api_url: str = "http://localhost:8001/api/v1/"
    flagsmith_server_key: str = ""

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Auth
    jwt_secret: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Rate limit
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    # Tokens
    max_input_tokens: int = 8000
    max_output_tokens: int = 2048
    max_context_tokens: int = 4000

    # Optional dedicated guard model (feature-flagged; empty disables)
    guard_model: str = ""
    guard_model_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`backend/app/core/logging.py`:
```python
import json
import logging
import sys
import time
from uuid import uuid4


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_correlation_id() -> str:
    return str(uuid4())
```

`backend/app/__init__.py` and `backend/app/core/__init__.py`: empty files.

`backend/.env.example`:
```dotenv
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5432/rag
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=documents
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
LLM_PRIMARY_MODEL=openai/gpt-4o
LLM_FALLBACK_MODEL=anthropic/claude-3-5-sonnet-20241022
LLM_API_KEY_PRIMARY=
LLM_API_KEY_FALLBACK=
EMBED_MODEL=openai/text-embedding-3-small
MAX_CONTEXT_TOKENS=4000
GUARD_MODEL=
GUARD_MODEL_ENABLED=false
RERANKER_PROVIDER=cohere
RERANKER_MODEL=rerank-english-v3.0
RERANKER_API_KEY=
FLAGSMITH_API_URL=http://localhost:8001/api/v1/
FLAGSMITH_SERVER_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
JWT_SECRET=change-me-in-prod
```

- [ ] **Step 4: Write the logging test**

`backend/tests/test_logging.py`:
```python
import json
import logging
import sys
import io

from app.core.logging import JsonFormatter, get_logger, new_correlation_id


def test_json_formatter_outputs_correlation_id():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test.json")
    logger.handlers = [handler]
    cid = new_correlation_id()
    logger = logging.LoggerAdapter(logger, {})  # placeholder to keep signature stable
    rec = logging.LogRecord("test.json", logging.INFO, __file__, 1, "hello", (), None)
    rec.correlation_id = cid
    handler.handle(rec)
    data = json.loads(stream.getvalue())
    assert data["msg"] == "hello"
    assert data["correlation_id"] == cid


def test_new_correlation_id_unique():
    assert new_correlation_id() != new_correlation_id()
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/ -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add backend
git commit -m "feat(backend): add pyproject, settings, and JSON logging"
```

---

### Task 0.2: docker-compose for all backing services

**Files:**
- Create: `infra/docker-compose.yml`
- Create: `infra/.env.example`
- Create: `infra/minio-init/init-bucket.sh`
- Create: `infra/prometheus/prometheus.yml`
- Create: `infra/grafana/provisioning/datasources/datasource.yml`
- Create: `infra/grafana/provisioning/dashboards/dashboards.yml`
- Create: `infra/langfuse/docker-compose.langfuse.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: container names, ports, credentials referenced by later phases:
  - Postgres `rag-postgres` on `:5432` (user/pass `rag/rag`, db `rag`).
  - MinIO `rag-minio` on `:9000`/`:9001`, console at `:9001`, access `minioadmin/minioadmin`, bucket `documents`.
  - Qdrant `rag-qdrant` on `:6333`/`:6334`.
  - Redis `rag-redis` on `:6379`.
  - RabbitMQ `rag-rabbitmq` on `:5672`/`:15672` (management UI).
  - Langfuse `rag-langfuse` on `:3000` (see langfuse compose).
  - Flagsmith `rag-flagsmith` on `:8000` (API) / `:8001` (edge) — edge is the SDK API at `:8001`.
  - Prometheus `rag-prometheus` on `:9090`.
  - Grafana `rag-grafana` on `:3001`.

- [ ] **Step 1: Write the main compose file**

`infra/docker-compose.yml`:
```yaml
version: "3.9"
name: rag-prod

services:
  postgres:
    image: postgres:16-alpine
    container_name: rag-postgres
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag"]
      interval: 5s
      timeout: 5s
      retries: 10

  minio:
    image: minio/minio:latest
    container_name: rag-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes:
      - "miniodata:/data"
      - "./minio-init:/docker-entrypoint-initdb.d"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 10

  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrantdata:/qdrant/storage"]
    healthcheck:
      test: ["CMD", "bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/6333 && echo ready"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 10

  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    container_name: rag-rabbitmq
    ports: ["5672:5672", "15672:15672"]
    volumes: ["rabbitmqdata:/var/lib/rabbitmq"]
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 5s
      timeout: 5s
      retries: 10

  flagsmith:
    image: flagsmith/flagsmith:latest
    container_name: rag-flagsmith
    environment:
      DATABASE_URL: postgres://rag:rag@postgres:5432/flagsmith
      EDGE_DATABASE_URL: postgres://rag:rag@postgres:5432/flag_engine
      FLAGSMITH_DATABASE_NAME: flagsmith
      FLAGSMITH_DATABASE_PASSWORD: rag
      FLAGSMITH_DATABASE_USER: rag
      DJANGO_ALLOWED_HOSTS: "*"
      ENABLE_EDGE: "true"
    ports: ["8000:8000"]
    depends_on:
      postgres:
        condition: service_healthy
    volumes: ["flagsmithdata:/var/lib/flagsmith"]

  edge:
    image: flagsmith/edge-proxy:latest
    container_name: rag-flagsmith-edge
    environment:
      DATABASE_URL: postgres://rag:rag@postgres:5432/flag_engine
    ports: ["8001:8000"]
    depends_on:
      postgres:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:latest
    container_name: rag-prometheus
    ports: ["9090:9090"]
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - "./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml"
    depends_on:
      - postgres

  grafana:
    image: grafana/grafana:latest
    container_name: rag-grafana
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
    ports: ["3001:3000"]
    volumes:
      - "./grafana/provisioning:/etc/grafana/provisioning"
      - "grafanadata:/var/lib/grafana"
    depends_on:
      - prometheus

volumes:
  pgdata:
  miniodata:
  qdrantdata:
  rabbitmqdata:
  flagsmithdata:
  grafanadata:
```

- [ ] **Step 2: Write minio-init script, prometheus, grafana configs**

`infra/minio-init/init-bucket.sh`:
```bash
#!/bin/sh
until mc alias set local http://minio:9000 minioadmin minioadmin; do sleep 2; done
mc mb --ignore-existing local/documents
mc version enable local/documents
mc anonymous set none local/documents
# lifecycle: expire non-current versions after 30 days
mc ilm rule add local/documents --expire-noncurrent-days 30 --noncurrent-expiration-newer-than 1
```

`infra/prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: "backend-api"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["host.docker.internal:8002"]
```

`infra/grafana/provisioning/datasources/datasource.yml`:
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

`infra/grafana/provisioning/dashboards/dashboards.yml`:
```yaml
apiVersion: 1
providers:
  - name: "default"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
```

`infra/langfuse/docker-compose.langfuse.yml` (no top-level `name:` — merged runs inherit the
main project `rag-prod`):
```yaml
services:
  langfuse:
    image: langfuse/langfuse:latest
    container_name: rag-langfuse
    environment:
      DATABASE_URL: postgresql://rag:rag@postgres:5432/langfuse
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: "change-me"
      SALT: "change-me"
      ENCRYPTION_KEY: "0000000000000000000000000000000000000000000000000000000000000000"
    ports: ["3000:3000"]
    depends_on:
      postgres:
        condition: service_healthy
```

- [ ] **Step 3: Verify compose files are valid**

Run: `docker compose -f infra/docker-compose.yml config -q`
Expected: exit 0, no output. (Langfuse compose validated separately via `-f infra/langfuse/docker-compose.langfuse.yml config -q`.)

- [ ] **Step 4: Commit**

```bash
git add infra
git commit -m "feat(infra): add docker-compose for all backing services"
```

---

### Task 0.3: FastAPI app skeleton with /health

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes_health.py`
- Test: `backend/tests/test_health.py`

**Interfaces:**
- Consumes: `get_settings()` from Task 0.1.
- Produces: FastAPI app factory `create_app() -> FastAPI` in `backend/app/main.py`; GET `/health`
  returning `{"status": "ok"}` — later phases mount real routers here.

- [ ] **Step 1: Write the failing health test**

`backend/tests/test_health.py`:
```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`backend/app/main.py`:
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    from app.api.routes_health import router as health_router

    app.include_router(health_router)
    return app
```

`backend/app/api/routes_health.py`:
```python
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    # Readiness checks (DB/Qdrant/Redis/Flagsmith/Langfuse) added in Phase 6
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/ -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app backend/tests/test_health.py
git commit -m "feat(backend): add FastAPI app skeleton with health route"
```

---

### Task 0.4: Shared typed errors

**Files:**
- Create: `backend/app/core/errors.py`
- Test: `backend/tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class DomainError(Exception)` with `.status_code: int` and `.detail: str`.
  - Subclasses used by later phases:
    - `AuthenticationError(status 401)`
    - `AuthorizationError(status 403)`
    - `ValidationError(status 422)`
    - `RateLimitError(status 429)`
    - `StorageError(status 500)`
    - `LLMError(status 502)`
    - `IngestionError(status 500)`
    - `NotFoundError(status 404)`
  - `register_exception_handlers(app: FastAPI) -> None` — maps `DomainError` (and subclasses) to
    JSON `{"error": detail, "trace_id": ...}`; always includes a `trace_id`.

- [ ] **Step 1: Write the failing error test**

`backend/tests/test_errors.py`:
```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import (
    DomainError,
    NotFoundError,
    register_exception_handlers,
)


def test_domain_error_status_and_detail():
    err = NotFoundError("missing")
    assert err.status_code == 404
    assert err.detail == "missing"


def test_register_exception_handlers_returns_json_with_trace_id():
    app = FastAPI()

    @app.get("/boom")
    async def boom():
        raise DomainError(500, "kaboom")

    register_exception_handlers(app)
    client = TestClient(app)
    resp = client.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "kaboom"
    assert "trace_id" in body
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest tests/test_errors.py -v`
Expected: FAIL.

- [ ] **Step 3: Write implementation**

`backend/app/core/errors.py`:
```python
from typing import TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger, new_correlation_id

logger = get_logger(__name__)


class DomainError(Exception):
    status_code = 500
    detail = "internal error"

    def __init__(self, status_code: int | None = None, detail: str | None = None) -> None:
        if status_code is not None:
            self.status_code = status_code
        if detail is not None:
            self.detail = detail
        super().__init__(self.detail)


class AuthenticationError(DomainError):
    status_code = 401
    detail = "authentication required"


class AuthorizationError(DomainError):
    status_code = 403
    detail = "forbidden"


class ValidationError(DomainError):
    status_code = 422
    detail = "validation failed"


class RateLimitError(DomainError):
    status_code = 429
    detail = "rate limit exceeded"


class StorageError(DomainError):
    status_code = 500
    detail = "storage error"


class LLMError(DomainError):
    status_code = 502
    detail = "llm error"


class IngestionError(DomainError):
    status_code = 500
    detail = "ingestion error"


class NotFoundError(DomainError):
    status_code = 404
    detail = "not found"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError):
        trace_id = new_correlation_id()
        logger.error("domain error", extra={"correlation_id": trace_id})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "trace_id": trace_id},
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python -m pytest tests/ -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/errors.py backend/tests/test_errors.py
git commit -m "feat(backend): add typed domain errors with trace_id handler"
```

---

**Phase 0 exit check:** `docker compose -f infra/docker-compose.yml up -d` starts all services;
`cd backend && python -m pytest tests/ -v` is green (7 tests); `/health` returns `{"status":"ok"}`.