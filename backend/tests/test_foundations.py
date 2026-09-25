from app.core.config import get_settings
from app.core.metrics import EVENTS_DROPPED
from app.models.document import DocumentStatus


def test_deleting_status_exists():
    assert DocumentStatus.DELETING.value == "DELETING"


def test_new_settings_defaults():
    s = get_settings()
    assert s.upload_max_bytes == 50 * 1024 * 1024
    assert s.presign_expiry_seconds == 900
    assert s.reaper_pending_after_seconds == 3600


def test_events_dropped_counter_has_reason_label():
    from prometheus_client import generate_latest

    EVENTS_DROPPED.labels(reason="test").inc()
    assert b"rag_events_dropped_total" in generate_latest()
    assert b'reason="test"' in generate_latest()
