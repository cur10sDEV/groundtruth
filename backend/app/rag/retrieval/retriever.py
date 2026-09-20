from dataclasses import dataclass
from functools import lru_cache

from app.core.qdrant_store import build_payload_filter, hybrid_search
from app.rag.embed.embeddings import dense_embed, sparse_embed


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    payload: dict


class Retriever:
    async def retrieve(
        self,
        query: str,
        org_id: str,
        user_ids: list[str],
        filters: dict,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        dense = (await dense_embed([query]))[0]
        sparse = sparse_embed([query])[0]
        payload_filter = build_payload_filter(org_id, user_ids, filters)
        points = hybrid_search(
            dense=dense,
            sparse_indices=sparse.indices,
            sparse_values=sparse.values,
            payload_filter=payload_filter,
            limit=limit,
        )
        out: list[RetrievedChunk] = []
        for p in points:
            out.append(
                RetrievedChunk(
                    chunk_id=p["id"],
                    text=p.get("payload", {}).get("chunk_text_hash", ""),
                    score=p.get("score", 0.0),
                    payload=p.get("payload", {}),
                )
            )
        return out


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
