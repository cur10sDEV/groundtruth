from dataclasses import dataclass
from functools import lru_cache

import litellm

from app.core.config import get_settings
from app.core.errors import LLMError


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


async def dense_embed(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    try:
        resp = await litellm.aembedding(
            model=s.embed_model,
            input=texts,
            api_key=s.llm_api_key_primary or None,
        )
        return [d["embedding"] for d in resp["data"]]
    except Exception as exc:
        raise LLMError(detail=f"dense embedding failed: {exc}") from exc


async def dense_embed_one(text: str) -> list[float]:
    return (await dense_embed([text]))[0]


@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


def sparse_embed(texts: list[str]) -> list[SparseVector]:
    model = _sparse_model()
    out: list[SparseVector] = []
    for vec in model.embed(texts):
        out.append(SparseVector(indices=[int(i) for i in vec.indices], values=vec.values))
    return out
