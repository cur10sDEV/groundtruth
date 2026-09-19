import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def fresh_settings():
    """Clear the lru_cache on get_settings so each test sees current env vars."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
