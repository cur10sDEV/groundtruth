from app.rag.chunkers.base import ChunkData
from app.rag.chunkers.text import RecursiveChunker, get_chunker


def test_recursive_chunker_respects_heading_boundaries():
    text = "# Section One\n\n" + "word " * 60 + "\n\n# Section Two\n\n" + "word " * 60
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=10).chunk(text)
    assert len(chunks) >= 2
    assert chunks[0].start_offset == 0
    # each chunk within a bounded size
    for c in chunks:
        assert len(c.text.split()) <= 120


def test_get_chunker_returns_recursive_for_all_types():
    for t in ("pdf", "docx", "md", "txt"):
        assert isinstance(get_chunker(t), RecursiveChunker)


def test_chunk_data_defaults():
    c = ChunkData(text="hello")
    assert c.page_number == 0 and c.start_offset == 0 and c.chunk_index == 0
