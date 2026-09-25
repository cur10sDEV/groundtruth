import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import setup_logging
from app.core.metrics import record_request


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.routes_auth import router as auth_router
    from app.api.routes_documents import router as documents_router
    from app.api.routes_health import router as health_router
    from app.api.routes_metrics import router as metrics_router
    from app.api.routes_query import router as query_router

    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(query_router)
    app.include_router(documents_router)
    app.include_router(metrics_router)
    register_exception_handlers(app)

    @app.middleware("http")
    async def metrics_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            record_request(time.perf_counter() - start, "error")
            raise
        status = "ok" if response.status_code < 500 else "error"
        record_request(time.perf_counter() - start, status)
        return response

    return app
