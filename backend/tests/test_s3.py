import pytest

from app.core.errors import StorageError
from app.core.s3 import (
    build_key,
    delete_object,
    get_object,
    key_to_parts,
    put_object,
)


def test_build_key_shape():
    key = build_key("org1", "user1", "doc1", "abc123.txt")
    assert key == "documents/org1/user1/doc1/abc123.txt"


def test_key_to_parts():
    org, user, doc, name = key_to_parts("documents/o/u/d/x.txt")
    assert (org, user, doc, name) == ("o", "u", "d", "x.txt")


@pytest.mark.integration
async def test_put_get_delete_roundtrip():
    key = put_object("org-i", "user-i", "doc-i", "hello.txt", b"hello world")
    assert get_object(key) == b"hello world"
    delete_object(key)
    with pytest.raises(StorageError):
        get_object(key)
