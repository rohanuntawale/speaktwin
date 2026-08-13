"""
Rate limiting, API-key auth, and request IDs.

These are exercised against a purpose-built app rather than the real one,
because the real app reads its middleware config from cached settings and
both features are deliberately off in the test environment.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)


def build_app(*, rate_limit: int | None = None, api_key: str | None = None):
    app = FastAPI()

    @app.get("/api/ping")
    async def ping():
        return {"ok": True}

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/other")
    async def other():
        return {"ok": True}

    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)
    if rate_limit is not None:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)
    app.add_middleware(RequestIDMiddleware)

    return TestClient(app)


# ---------------------------------------------------------------------------
# Request IDs
# ---------------------------------------------------------------------------
def test_request_id_is_generated():
    response = build_app().get("/api/ping")
    assert response.headers["X-Request-ID"]


def test_caller_supplied_request_id_is_preserved():
    client = build_app()
    response = client.get("/api/ping", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_requests_are_allowed_up_to_the_limit():
    client = build_app(rate_limit=3)
    assert [client.get("/api/ping").status_code for _ in range(3)] == [200, 200, 200]


def test_requests_beyond_the_limit_are_rejected():
    client = build_app(rate_limit=2)
    for _ in range(2):
        client.get("/api/ping")

    response = client.get("/api/ping")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    assert response.json()["status"] == "error"


def test_rate_limiting_ignores_non_api_paths():
    client = build_app(rate_limit=1)
    client.get("/api/ping")
    assert client.get("/other").status_code == 200


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
def test_api_requires_the_key_when_configured():
    client = build_app(api_key="secret")
    assert client.get("/api/ping").status_code == 401
    assert client.get("/api/ping", headers={"X-API-Key": "wrong"}).status_code == 401


def test_correct_key_is_accepted():
    client = build_app(api_key="secret")
    assert client.get("/api/ping", headers={"X-API-Key": "secret"}).status_code == 200


def test_health_stays_reachable_without_a_key():
    """Load balancers probe health unauthenticated."""
    assert build_app(api_key="secret").get("/api/health").status_code == 200


def test_non_api_paths_are_not_gated():
    """The frontend itself must still load so it can prompt for a key."""
    assert build_app(api_key="secret").get("/other").status_code == 200
