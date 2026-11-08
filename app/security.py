"""API-key auth and an in-process sliding-window rate limiter."""
from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import Settings


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            retry = int(60 - (now - q[0])) + 1
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(retry)})
        q.append(now)


def make_guard(s: Settings, limiter: RateLimiter):
    expected = s.secret("api_auth_key")

    async def guard(request: Request) -> None:
        if expected:
            supplied = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                raise HTTPException(401, "Invalid or missing X-API-Key")
        client = request.headers.get("x-api-key") or (request.client.host if request.client else "anon")
        limiter.check(client)

    return guard
