from __future__ import annotations

import json
import time

import redis.asyncio as aioredis

from app.core.config import get_settings


class RedisCache:
    def __init__(self, url: str) -> None:
        self._r = aioredis.from_url(url, decode_responses=True)

    async def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        await self._r.set(key, json.dumps(value), ex=ttl)

    async def get(self, key: str) -> dict | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw else None

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def keys_by_pattern(self, pattern: str) -> list[str]:
        return [k async for k in self._r.scan_iter(match=pattern)]

    async def delete_by_pattern(self, pattern: str) -> int:
        keys = [k async for k in self._r.scan_iter(match=pattern)]
        if keys:
            return await self._r.delete(*keys)
        return 0

    async def sadd(self, key: str, *members: str) -> None:
        await self._r.sadd(key, *members)

    async def smembers(self, key: str) -> set[str]:
        return set(await self._r.smembers(key))


class RateLimiter:
    def __init__(self, url: str) -> None:
        self._r = aioredis.from_url(url, decode_responses=True)

    async def allow(self, user_key: str, limit: int, window_seconds: int) -> bool:
        now = int(time.time())
        window = now // window_seconds
        k = f"rl:{user_key}:{window}"
        count = await self._r.incr(k)
        if count == 1:
            await self._r.expire(k, window_seconds)
        return count <= limit


_cache: RedisCache | None = None
_limiter: RateLimiter | None = None


def get_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache(get_settings().redis_url)
    return _cache


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().redis_url)
    return _limiter
