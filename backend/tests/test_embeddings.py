import litellm
import pytest

from app.core.errors import LLMError
from app.rag.embed.embeddings import SparseVector, dense_embed, sparse_embed


def test_sparse_vector_dataclass():
    sv = SparseVector(indices=[1, 5], values=[0.5, 0.3])
    assert len(sv.indices) == 2 and len(sv.values) == 2


def test_sparse_embed_shape():
    results = sparse_embed(["hello world", "rag system"])
    assert len(results) == 2
    for r in results:
        assert len(r.indices) == len(r.values)
        assert len(r.indices) > 0


async def test_dense_embed_maps_response(monkeypatch):
    captured = {}

    async def fake_aembedding(**kwargs):
        captured.update(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)
    result = await dense_embed(["alpha", "beta"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["input"] == ["alpha", "beta"]
    assert captured["model"] == "openai/text-embedding-3-small"


async def test_dense_embed_raises_llm_error_on_failure(monkeypatch):
    async def failing_aembedding(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(litellm, "aembedding", failing_aembedding)
    with pytest.raises(LLMError):
        await dense_embed(["alpha"])
