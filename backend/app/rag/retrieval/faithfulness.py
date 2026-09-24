import json
from dataclasses import dataclass

import litellm

from app.core.config import get_settings
from app.core.errors import LLMError
from app.rag.retrieval.generate import build_context_block


@dataclass
class FaithfulnessResult:
    faithful: bool
    score: float


async def check_faithfulness(query: str, answer: str, contexts: list[dict]) -> FaithfulnessResult:
    s = get_settings()
    block = build_context_block(contexts)
    prompt = (
        "On a scale 0.0 to 1.0, how well is the following answer supported ONLY by the "
        "context passages? Consider any claim not present in the context as unsupported. "
        'Return JSON {"score": <float>}.\n\nContext:\n'
        f"{block}\n\nQuestion: {query}\nAnswer: {answer}"
    )
    attempts: list[tuple[str, str]] = [(s.llm_primary_model, s.llm_api_key_primary)]
    if s.llm_fallback_model:
        attempts.append((s.llm_fallback_model, s.llm_api_key_fallback))
    last_exc: Exception | None = None
    for model, api_key in attempts:
        try:
            resp = await litellm.acompletion(
                model=model,
                api_key=api_key or None,
                messages=[
                    {"role": "system", "content": "You are a strict fact-checking judge."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content)
            score = float(data.get("score", 0.0))
            return FaithfulnessResult(faithful=score >= 0.7, score=score)
        except Exception as exc:
            last_exc = exc
    raise LLMError(detail=f"faithfulness check failed: {last_exc}") from last_exc
