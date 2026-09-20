from app.rag.retrieval.retriever import RetrievedChunk, get_retriever


def test_retrieved_chunk_defaults():
    rc = RetrievedChunk(chunk_id="c1", text="hi", score=0.5, payload={})
    assert rc.chunk_id == "c1"
    assert rc.score == 0.5


def test_get_retriever_singleton():
    assert get_retriever() is get_retriever()
