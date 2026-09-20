import pytest

from app.core.redis_store import get_cache, get_limiter


@pytest.mark.integration
async def test_cache_set_get_delete():
    cache = get_cache()
    await cache.set("k", {"a": 1})
    assert (await cache.get("k")) == {"a": 1}
    await cache.delete("k")
    assert (await cache.get("k")) is None


@pytest.mark.integration
async def test_rate_limiter_allows_then_blocks():
    limiter = get_limiter()
    assert await limiter.allow("u1", limit=2, window_seconds=60)
    assert await limiter.allow("u1", limit=2, window_seconds=60)
    assert not await limiter.allow("u1", limit=2, window_seconds=60)
