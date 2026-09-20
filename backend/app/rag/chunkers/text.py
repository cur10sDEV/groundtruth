import re

from app.rag.chunkers.base import BaseChunker, ChunkData

_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _split_heads(text: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in _HEADING_RE.finditer(text)]
    bounds = [0] + [e for _, e in spans] + [len(text)]
    out = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e > s:
            out.append((s, e))
    return out


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, page_number: int = 0) -> list[ChunkData]:
        chunks: list[ChunkData] = []
        for s, e in _split_heads(text):
            section = text[s:e]
            for para_start, para_end in _split_paragraphs(section):
                para = section[para_start:para_end]
                for tok_start, tok_end in _split_by_words(
                    para, self.chunk_size, self.chunk_overlap
                ):
                    chunks.append(
                        ChunkData(
                            text=para[tok_start:tok_end].strip(),
                            page_number=page_number,
                            start_offset=s + para_start + tok_start,
                            end_offset=s + para_start + tok_end,
                            chunk_index=len(chunks),
                        )
                    )
        if not chunks:
            chunks.append(ChunkData(text=text.strip(), page_number=page_number, chunk_index=0))
        return chunks


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    matches = [m.span() for m in _PARAGRAPH_RE.finditer(text)]
    starts = [0] + [e for _, e in matches]
    ends = [s for s, _ in matches] + [len(text)]
    return [(starts[i], ends[i]) for i in range(len(starts)) if ends[i] > starts[i]]


def _split_by_words(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    bounds = [w.span() for w in words]
    spans: list[tuple[int, int]] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(bounds):
        end = min(i + size, len(bounds))
        spans.append((bounds[i][0], bounds[end - 1][1]))
        i += step
    return spans


def get_chunker(doc_type: str, chunk_size: int = 400, chunk_overlap: int = 80) -> BaseChunker:
    return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
