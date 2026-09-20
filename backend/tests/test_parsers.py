import pytest

from app.core.errors import IngestionError
from app.rag.ingestion.parsers import (
    ParsedDocument,
    parse_bytes,
    parse_markdown,
    parse_txt,
)


def test_parse_txt():
    doc = parse_txt(b"hello\nworld")
    assert doc.full_text == "hello\nworld"
    assert len(doc.pages) == 1


def test_parse_markdown():
    doc = parse_markdown(b"# Title\n\nbody")
    assert "# Title" in doc.full_text


def test_parse_bytes_unknown_type():
    with pytest.raises(IngestionError):
        parse_bytes("file.xyz", b"data")


def test_parse_bytes_txt():
    doc = parse_bytes("notes.txt", b"some text")
    assert isinstance(doc, ParsedDocument)
    assert "some text" in doc.full_text
