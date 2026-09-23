import cohere

from app.core.config import get_settings
from app.core.errors import DomainError
from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankerDisabledError(DomainError):
    status_code = 503
    detail = "reranker is disabled or not configured"


async def rerank(query: str, contexts: list[dict], top_n: int = 5) -> list[dict]:
    s = get_settings()
    if not s.reranker_api_key:
        raise RerankerDisabledError()
    try:
        client = cohere.Client(s.reranker_api_key)
        if s.reranker_provider == "cohere":
            docs = [c["text"] for c in contexts]
            resp = client.rerank(
                model=s.reranker_model,
                query=query,
                documents=docs,
                top_n=top_n,
            )
            ordered = [contexts[r.index] for r in resp.results]
            return ordered
        raise RerankerDisabledError()
    except RerankerDisabledError:
        raise
    except Exception as exc:
        logger.warning("rerank failed, returning original order", extra={"exc": str(exc)})
        return contexts
