from dataclasses import dataclass


@dataclass
class ChunkData:
    text: str
    page_number: int = 0
    start_offset: int = 0
    end_offset: int = 0
    chunk_index: int = 0


class BaseChunker:
    def chunk(self, text: str, page_number: int = 0) -> list[ChunkData]:
        raise NotImplementedError
