from collections.abc import AsyncIterator

import litellm

from app.core.config import get_settings
from app.core.errors import LLMError


def build_context_block(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))


def truncate_contexts(contexts: list[dict], max_tokens: int) -> list[dict]:
    budget = max_tokens
    out: list[dict] = []
    for c in contexts:
        tokens = len(c["text"]) // 4 + 1
        if tokens > budget:
            break
        budget -= tokens
        out.append(c)
    return out


async def generate_answer(contexts: list[dict], query: str) -> AsyncIterator[dict]:
    s = get_settings()
    system = (
        "You are a grounded question-answering assistant. Answer ONLY using the provided "
        "numbered context passages. Cite sources inline as [n]. If the answer is not in the "
        "context, reply exactly: 'I cannot confidently answer that based on the available "
        "documents.' Do not fabricate facts."
    )
    block = build_context_block(contexts)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{block}\n\nQuestion: {query}"},
    ]
    try:
        resp = await litellm.acompletion(
            model=s.llm_primary_model,
            api_key=s.llm_api_key_primary or None,
            messages=messages,
            max_tokens=s.max_output_tokens,
            stream=True,
        )
        model_used = s.llm_primary_model
        first = True
        async for chunk in resp:
            if first and getattr(chunk, "model", None):
                model_used = chunk.model
                yield {"type": "meta", "model_used": model_used}
                first = False
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield {"type": "token", "text": delta}
        if first:
            yield {"type": "meta", "model_used": model_used}
    except Exception as exc:
        raise LLMError(detail=f"generation failed: {exc}") from exc
