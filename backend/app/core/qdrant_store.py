from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.core.errors import StorageError


def _client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)


def ensure_collection() -> None:
    s = get_settings()
    client = _client()
    try:
        exists = client.collection_exists(s.qdrant_collection)
        if not exists:
            client.create_collection(
                collection_name=s.qdrant_collection,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=s.embed_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse_vector": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
    except Exception as exc:
        raise StorageError(detail=f"qdrant ensure_collection failed: {exc}") from exc
    finally:
        client.close()


def upsert_points_batch(points: list[dict]) -> None:
    s = get_settings()
    client = _client()
    try:
        structs = [
            models.PointStruct(
                id=p["id"],
                vector={
                    "dense_vector": p["dense"],
                    "bm25_sparse_vector": models.SparseVector(
                        indices=p["sparse_indices"], values=p["sparse_values"]
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ]
        client.upsert(collection_name=s.qdrant_collection, points=structs)
    except Exception as exc:
        raise StorageError(detail=f"qdrant upsert failed: {exc}") from exc
    finally:
        client.close()


def upsert_point(
    point_id: str,
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload: dict,
) -> None:
    upsert_points_batch(
        [
            {
                "id": point_id,
                "dense": dense,
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values,
                "payload": payload,
            }
        ]
    )


def hybrid_search(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    payload_filter: dict,
    limit: int = 10,
) -> list[dict]:
    s = get_settings()
    client = _client()
    try:
        result = client.query_points(
            collection_name=s.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense,
                    using="dense_vector",
                    limit=limit * 2,
                    filter=models.Filter(**payload_filter) if payload_filter else None,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using="bm25_sparse_vector",
                    limit=limit * 2,
                    filter=models.Filter(**payload_filter) if payload_filter else None,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [point.model_dump() for point in result.points]
    except Exception as exc:
        raise StorageError(detail=f"qdrant search failed: {exc}") from exc
    finally:
        client.close()


def delete_points(point_ids: list[str], payload_filter: dict | None = None) -> None:
    s = get_settings()
    client = _client()
    try:
        if point_ids:
            client.delete(collection_name=s.qdrant_collection, points_selector=point_ids)
        elif payload_filter:
            client.delete(
                collection_name=s.qdrant_collection,
                points_selector=models.FilterSelector(filter=models.Filter(**payload_filter)),
            )
    except Exception as exc:
        raise StorageError(detail=f"qdrant delete failed: {exc}") from exc
    finally:
        client.close()


def build_payload_filter(org_id: str, user_ids: list[str] | None, extra: dict) -> dict:
    must: list[dict] = [{"key": "org_id", "match": {"value": org_id}}]
    if user_ids:
        must.append({"key": "user_id", "match": {"any": user_ids}})
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            must.append({"key": k, "match": {"any": list(v)}})
        else:
            must.append({"key": k, "match": {"value": v}})
    return {"must": must}
