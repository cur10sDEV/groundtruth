import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.auth.dependencies import get_current_user
from app.core.config import get_settings
from app.core.errors import RateLimitError, ValidationError
from app.core.logging import new_correlation_id
from app.core.redis_store import get_limiter
from app.db import get_session
from app.models.chunk import Chunk
from app.models.citation import Citation
from app.rag.retrieval.orchestrator import run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryIn(BaseModel):
    query: str


@router.post("")
async def query_endpoint(
    body: QueryIn,
    user: dict = Depends(get_current_user),
):
    settings = get_settings()
    limiter = get_limiter()
    allowed = await limiter.allow(
        f"user:{user['user_id']}",
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )
    if not allowed:
        raise RateLimitError()

    # token budget guard: approx 4 chars/token; refuses absurdly large queries pre-stream
    if len(body.query) // 4 + 1 > settings.max_input_tokens:
        raise ValidationError(detail="query exceeds input token budget")

    trace_id = new_correlation_id()

    async def event_stream():
        async for ev in run_query(
            body.query,
            user["org_id"],
            [user["user_id"]],
            feature_flags={"cache.enabled": True, "faithfulness.enabled": True},
            trace_id=trace_id,
        ):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{query_id}/citations")
async def citations_endpoint(
    query_id: str,
    user: dict = Depends(get_current_user),
) -> list[dict]:
    async with get_session() as session:
        cited = (
            (await session.execute(select(Citation).where(Citation.query_id == query_id)))
            .scalars()
            .all()
        )
        if not cited:
            return []
        chunk_ids = [c.chunk_id for c in cited]
        chunks = (
            (
                await session.execute(
                    select(Chunk).where(Chunk.id.in_(chunk_ids), Chunk.org_id == user["org_id"])
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "chunk_id": str(c.id),
                "doc_id": c.doc_id,
                "text": c.chunk_text,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
            }
            for c in chunks
        ]
