import pytest

from app.core import s3 as s3mod
from app.core.config import get_settings
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


class _FakePresignClient:
    def generate_presigned_post(self, Bucket, Key, Conditions, ExpiresIn):
        self.calls = {"bucket": Bucket, "key": Key, "conditions": Conditions, "expires": ExpiresIn}
        return {"url": "http://minio/documents", "fields": {"key": Key, "x": "1"}}


class _FakeListClient:
    def __init__(self, keys):
        self.keys = keys
        self.deleted = []

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.keys if k.startswith(Prefix)]}

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.deleted.append(obj["Key"])
        return {"Deleted": Delete["Objects"]}


def test_presign_upload_shape(monkeypatch):
    fake = _FakePresignClient()
    monkeypatch.setattr(s3mod, "_client", lambda: fake)
    out = s3mod.presign_upload("o", "u", "d", "pdf")
    assert out["url"] == "http://minio/documents"
    assert out["fields"]["key"].startswith("documents/o/u/d/")
    assert out["fields"]["key"].endswith(".pdf")
    assert fake.calls["conditions"] == [
        ["content-length-range", 0, get_settings().upload_max_bytes]
    ]
    assert fake.calls["expires"] == get_settings().presign_expiry_seconds


def test_delete_prefix_deletes_all_versions(monkeypatch):
    fake = _FakeListClient(
        ["documents/o/u/d/a.pdf", "documents/o/u/d/b.pdf", "documents/o/u/e/c.pdf"]
    )
    monkeypatch.setattr(s3mod, "_client", lambda: fake)
    assert s3mod.delete_prefix("o", "u", "d") == 2
    assert sorted(fake.deleted) == ["documents/o/u/d/a.pdf", "documents/o/u/d/b.pdf"]


def test_delete_prefix_empty_is_noop(monkeypatch):
    fake = _FakeListClient([])
    monkeypatch.setattr(s3mod, "_client", lambda: fake)
    assert s3mod.delete_prefix("o", "u", "d") == 0
    assert fake.deleted == []  # delete_objects never called: no batches
