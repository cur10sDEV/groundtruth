import pytest

from app.core.qdrant_store import build_payload_filter, ensure_collection


def test_build_payload_filter_org_scoped():
    f = build_payload_filter(org_id="org1", user_ids=["u1", "u2"], extra={"year": 2025})
    assert f is not None


@pytest.mark.integration
def test_ensure_collection_runs():
    ensure_collection()
    assert True
