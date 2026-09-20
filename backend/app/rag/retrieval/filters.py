import json
from dataclasses import dataclass

import litellm

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

FILTERABLE_FIELDS: dict[str, type] = {
    "year": int,
    "type": str,
    "tags": list[str],
    "topic": str,
    "page_number": int,
}


@dataclass
class Filters:
    year: int | None = None
    type: str | None = None
    tags: list[str] | None = None
    topic: str | None = None
    page_number: int | None = None

    def to_payload(self) -> dict:
        out: dict = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            out[k] = v
        return out


def default_filters() -> Filters:
    return Filters()


def _extract_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "year": {"type": "integer"},
            "type": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "topic": {"type": "string"},
            "page_number": {"type": "integer"},
        },
        "additionalProperties": False,
    }


async def extract_filters(query: str) -> Filters:
    s = get_settings()
    system = (
        "You extract metadata filters for a document search from a user query. "
        "Return ONLY JSON matching the schema. Use null for fields not implied by the query."
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
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "apply_filters",
                        "description": "Apply metadata filters to the search",
                        "parameters": _extract_schema(),
                    },
                }
            ],
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        content = None
        if tool_calls:
            content = tool_calls[0].function.arguments
        elif msg.content:
            content = msg.content
        if not content:
            return default_filters()
        data = json.loads(content)
        return Filters(
            year=data.get("year"),
            type=data.get("type"),
            tags=data.get("tags"),
            topic=data.get("topic"),
            page_number=data.get("page_number"),
        )
    except Exception as exc:
        logger.warning("filter extraction failed, using defaults", extra={"exc": str(exc)})
        return default_filters()
