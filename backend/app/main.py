from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    from app.api.routes_health import router as health_router
    from app.ingestion.minio_webhook import register_minio_webhook

    app.include_router(health_router)
    register_minio_webhook(app)
    register_exception_handlers(app)
    return app
