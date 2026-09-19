import io
import json
import logging

from app.core.logging import JsonFormatter, new_correlation_id


def test_json_formatter_outputs_correlation_id():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test.json")
    logger.handlers = [handler]
    cid = new_correlation_id()
    logger = logging.LoggerAdapter(logger, {})  # placeholder to keep signature stable
    rec = logging.LogRecord("test.json", logging.INFO, __file__, 1, "hello", (), None)
    rec.correlation_id = cid
    handler.handle(rec)
    data = json.loads(stream.getvalue())
    assert data["msg"] == "hello"
    assert data["correlation_id"] == cid


def test_new_correlation_id_unique():
    assert new_correlation_id() != new_correlation_id()
