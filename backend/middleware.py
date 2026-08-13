"""
SpeakTwin - HTTP Middleware
============================
Cross-cutting request concerns:

  * RequestIDMiddleware - tags every request so a log line can be traced
    back to the response the caller saw
  * RateLimitMiddleware - a per-client token bucket, because /api/analyze
    can spend money at a cloud provider on every call
  * APIKeyMiddleware - optional shared-secret gate, off unless API_KEY is set

The rate limiter is in-process. That is the right fit for a single-worker
deployment; running several workers behind a load balancer would need a
shared backend such as Redis.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from typing import Dict, Tuple

from fastapi import Request  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore

from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import get_logger  # type: ignore

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"

# Readable by any code handling the request (used by the error handlers).
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

# Paths that must stay reachable without a key, so a load balancer can
# health-check the service.
PUBLIC_PATHS = frozenset({"/api/health"})


def current_request_id() -> str:
    return request_id_ctx.get()


def _client_key(request: Request) -> str:
    """Identify the caller for rate limiting."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _error(status_code: int, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"message": message, "status": "error", "request_id": request_id},
        status_code=status_code,
        headers={REQUEST_ID_HEADER: request_id},
    )


# ---------------------------------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, echo it back, and log slow API calls."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id

        if request.url.path.startswith("/api"):
            logger.info(
                "%s %s -> %d (%.0f ms) [%s]",
                request.method, request.url.path, response.status_code,
                elapsed_ms, request_id,
            )
        return response


# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket per client, applied to /api routes only."""

    def __init__(self, app, requests_per_minute: int):
        super().__init__(app)
        self.capacity = float(max(1, requests_per_minute))
        self.refill_per_second = self.capacity / 60.0
        # client -> (tokens, last_refill_monotonic)
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._last_prune = time.monotonic()

    def _prune(self, now: float) -> None:
        if now - self._last_prune < 300:
            return
        self._last_prune = now
        stale = [k for k, (_, seen) in self._buckets.items() if now - seen > 600]
        for key in stale:
            self._buckets.pop(key, None)

    def _allow(self, key: str) -> bool:
        now = time.monotonic()
        self._prune(now)

        tokens, last = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)

        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False

        self._buckets[key] = (tokens - 1.0, now)
        return True

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        key = _client_key(request)
        if not self._allow(key):
            request_id = getattr(request.state, "request_id", "-")
            logger.warning("Rate limit hit for %s [%s]", key, request_id)
            response = _error(429, "Too many requests. Please slow down.", request_id)
            response.headers["Retry-After"] = "5"
            return response

        return await call_next(request)


# ---------------------------------------------------------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require `X-API-Key` on /api routes when a key is configured."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api") or path in PUBLIC_PATHS:
            return await call_next(request)

        # Browsers send an unauthenticated OPTIONS preflight by design.
        if request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get(API_KEY_HEADER, "")
        if not _constant_time_equals(provided, self.api_key):
            request_id = getattr(request.state, "request_id", "-")
            return _error(401, "Invalid or missing API key.", request_id)

        return await call_next(request)


def _constant_time_equals(left: str, right: str) -> bool:
    """Compare without leaking length or position through timing."""
    import hmac
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


# ---------------------------------------------------------------------------
def install(app) -> None:
    """
    Register middleware in the correct order.

    Starlette runs the most recently added middleware first, so these are
    added inside-out: the CORS layer registered last in main.py ends up
    outermost and can decorate even a 401 or 429.
    """
    settings = get_settings()

    if settings.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
        logger.info("API key authentication enabled")

    if settings.rate_limit_per_minute > 0:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_per_minute,
        )
        logger.info("Rate limiting enabled (%d req/min per client)",
                    settings.rate_limit_per_minute)

    app.add_middleware(RequestIDMiddleware)
