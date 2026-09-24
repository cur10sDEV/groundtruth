from fastapi.testclient import TestClient

from app.main import create_app


def test_metrics_endpoint():
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_requests_total" in resp.text
