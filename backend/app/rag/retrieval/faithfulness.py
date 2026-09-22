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
    try:
        resp = await litellm.acompletion(
            model=s.llm_primary_model,
            api_key=s.llm_api_key_primary or None,
            messages=[
                {"role": "system", "content": "You are a strict fact-checking judge."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        score = float(data.get("score", 0.0))
    except Exception as exc:
        raise LLMError(detail=f"faithfulness check failed: {exc}") from exc
    return FaithfulnessResult(faithful=score >= 0.7, score=score)
