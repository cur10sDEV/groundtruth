import pytest

import app.core.redis_store as redis_store
from app.core.redis_store import get_cache, get_limiter


@pytest.fixture(autouse=True)
def fresh_redis_singletons():
    """Reset module singletons so each test gets a client bound to its own event loop."""
    redis_store._cache = None
    redis_store._limiter = None
    yield
    redis_store._cache = None
    redis_store._limiter = None


@pytest.mark.integration
async def test_cache_set_get_delete():
    cache = get_cache()
    await cache.set("k", {"a": 1})
    assert (await cache.get("k")) == {"a": 1}
    await cache.delete("k")
    assert (await cache.get("k")) is None


@pytest.mark.integration
async def test_cache_keys_by_pattern():
    cache = get_cache()
    await cache.set("kbp:a", {"x": 1})
    await cache.set("kbp:b", {"x": 2})
    await cache.set("kbp:other-org", {"x": 3})
    await cache.set("unrelated", {"x": 4})
    try:
        assert sorted(await cache.keys_by_pattern("kbp:*")) == ["kbp:a", "kbp:b", "kbp:other-org"]
        assert await cache.keys_by_pattern("kbp:a") == ["kbp:a"]
        assert await cache.keys_by_pattern("nomatch:*") == []
    finally:
        for k in ("kbp:a", "kbp:b", "kbp:other-org", "unrelated"):
            await cache.delete(k)


@pytest.mark.integration
async def test_rate_limiter_allows_then_blocks():
    limiter = get_limiter()
    assert await limiter.allow("u1", limit=2, window_seconds=60)
    assert await limiter.allow("u1", limit=2, window_seconds=60)
    assert not await limiter.allow("u1", limit=2, window_seconds=60)
