"""Phase 8.2: API Rate Limiting and Idempotency Middleware.

Ensures clients are rate-limited and batch transaction ingestions are idempotent.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, Response, status
import structlog

logger = structlog.get_logger(__name__)

# Global cache for idempotency: mapping key -> (status_code, body_bytes)
IDEMPOTENCY_CACHE: dict[str, tuple[int, bytes]] = {}

# In-memory rate limiting fallback: mapping client_ip -> list of timestamps
RATE_LIMIT_CACHE: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100


async def check_idempotency(request: Request) -> str | None:
    """FastAPI dependency to verify idempotency keys."""
    key = request.headers.get("X-Idempotency-Key")
    if not key:
        return None

    if key in IDEMPOTENCY_CACHE:
        status_code, body = IDEMPOTENCY_CACHE[key]
        logger.info("Cache hit for idempotency key", key=key)
        # Fast-exit with cached response using a custom HTTPException
        # or we handle this directly in the endpoint.
        # To make it clean, we let endpoints check and return.
    return key


def cache_idempotency_response(key: str, status_code: int, body: bytes) -> None:
    """Store the response content in the idempotency cache."""
    IDEMPOTENCY_CACHE[key] = (status_code, body)


async def rate_limiter(request: Request) -> None:
    """Enforce rate limits on incoming API requests."""
    client_ip = request.client.host if request.client else "unknown_ip"
    now = time.time()

    # Clear old requests outside window
    RATE_LIMIT_CACHE[client_ip] = [
        t for t in RATE_LIMIT_CACHE[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]

    if len(RATE_LIMIT_CACHE[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("Rate limit exceeded for client", ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 100 requests per minute.",
        )

    RATE_LIMIT_CACHE[client_ip].append(now)
