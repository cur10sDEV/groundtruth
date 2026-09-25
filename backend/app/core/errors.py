from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_correlation_id, get_logger, new_correlation_id

logger = get_logger(__name__)


class DomainError(Exception):
    status_code = 500
    detail = "internal error"

    def __init__(self, status_code: int | None = None, detail: str | None = None) -> None:
        if detail is None and isinstance(status_code, str):
            status_code, detail = None, status_code
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
        # reuse the request's correlation id when one is in scope (set by the
        # query routes) instead of minting a fresh one
        trace_id = get_correlation_id() or new_correlation_id()
        logger.error("domain error", extra={"correlation_id": trace_id})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "trace_id": trace_id},
        )
