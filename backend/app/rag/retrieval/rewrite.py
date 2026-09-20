import json
from dataclasses import dataclass

import litellm

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RewrittenQueries:
    canonical: str
    queries: list[str]


def fallback_rewritten(query: str) -> RewrittenQueries:
    return RewrittenQueries(canonical=query, queries=[query])


async def rewrite_query(query: str, k: int = 3) -> RewrittenQueries:
    s = get_settings()
    system = (
        "Fix typos and grammar, keep the original intent unchanged, then produce "
        f"{k} distinct paraphrases that improve retrieval recall. Return JSON with "
        'keys "canonical" (the cleaned query) and "queries" (list of paraphrases).'
    )
    try:
        resp = await litellm.acompletion(
            model=s.llm_primary_model,
            api_key=s.llm_api_key_primary or None,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        canonical = data.get("canonical") or query
        queries = data.get("queries") or [query]
        return RewrittenQueries(canonical=canonical, queries=[canonical, *queries])
    except Exception as exc:
        logger.warning("query rewrite failed, using original", extra={"exc": str(exc)})
        return fallback_rewritten(query)
