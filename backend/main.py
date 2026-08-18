"""
SpeakTwin - AI Communication Mirror
====================================
FastAPI application entry point.

  1. Builds the app with configured CORS, request IDs, rate limiting, and
     optional API-key auth
  2. Registers the API routes
  3. Pre-loads the STT model at startup so the first chunk is not slow
  4. Serves the frontend

Static-file note: the frontend is mounted at BOTH `/static` and `/`. The
root mount is what makes `index.html` work, because the page loads
`style.css` and `app.js` with relative paths - with only the `/static`
mount those resolved to `/style.css` and 404'd. The mount is registered
after the API router so `/api/*` still wins.
"""

from __future__ import annotations

import os
import sys
import traceback

# Running this file directly (`python backend/main.py`) puts `backend/` on
# sys.path rather than the project root, so `from backend import ...` cannot
# resolve and the app dies before it starts. Put the root on the path first
# so the file works both as a script and as `uvicorn backend.main:app`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request  # type: ignore
from fastapi.exceptions import RequestValidationError  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from contextlib import asynccontextmanager

from backend import middleware as mw  # type: ignore
from backend.routes.analyze import router as analyze_router  # type: ignore
from backend.schemas import HealthResponse  # type: ignore
from backend.services.session_store import session_store  # type: ignore
from backend.utils.config import FRONTEND_DIR, get_settings  # type: ignore
from backend.utils.helpers import get_logger  # type: ignore

logger = get_logger("speaktwin")
settings = get_settings()

APP_VERSION = "1.2.0"


# ---------------------------------------------------------------------------
# Lifespan - model pre-loading on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up expensive resources before the first request arrives."""
    logger.info("=" * 60)
    logger.info("  SpeakTwin - AI Communication Mirror  v%s", APP_VERSION)
    logger.info("  STT engine: %s | LLM insight: %s",
                settings.resolved_stt_engine,
                "on" if settings.llm_enabled else "off")
    logger.info("=" * 60)

    # The local model is the fallback for every engine, so warm it unless a
    # cloud engine is explicitly configured as the only path.
    if settings.resolved_stt_engine == "local":
        try:
            from backend.services.speech_to_text import _load_model  # type: ignore
            if _load_model() is not None:
                logger.info("Local STT model ready")
            else:
                logger.warning(
                    "Local STT model unavailable - transcription will fail "
                    "until GROQ_API_KEY or OPENAI_API_KEY is configured"
                )
        except Exception as exc:
            logger.warning("Could not pre-load the STT model: %s", exc)
    else:
        logger.info("Cloud STT configured; skipping local model pre-load")

    # Warm the neural models so the first chunk does not pay the load cost.
    if settings.any_ml_enabled and settings.warmup_models:
        try:
            from backend.services.ml import enrichment  # noqa: F401  # type: ignore
            from backend.services.ml.registry import registry  # type: ignore

            logger.info("Warming ML models: %s", ", ".join(settings.warmup_models))
            registry.warmup(settings.warmup_models)
            for model in registry.status()["models"]:
                if model["load_attempted"] and not model["loaded"]:
                    logger.warning("  %s unavailable: %s", model["key"], model["error"])
        except Exception as exc:
            logger.warning("ML warmup skipped: %s", exc)

    yield

    logger.info("SpeakTwin shutting down...")
    session_store.clear()


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SpeakTwin - AI Communication Mirror",
    description="Real-time speech analysis and coaching feedback",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Request ID -> rate limit -> API key. Added first so CORS ends up outermost.
mw.install(app)

# CORS. Credentials are only allowed alongside explicit origins: the spec
# forbids pairing `Access-Control-Allow-Credentials: true` with `*`, and
# browsers reject the combination outright.
_wildcard_cors = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not _wildcard_cors,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[mw.REQUEST_ID_HEADER],
)
if _wildcard_cors:
    logger.warning(
        "CORS is open to all origins. Set CORS_ORIGINS to your real "
        "front-end origin before deploying."
    )


# ---------------------------------------------------------------------------
# Error handlers - one response shape, no internals leaked
# ---------------------------------------------------------------------------
def _error_body(message: str, detail: str | None = None) -> dict:
    body = {
        "message": message,
        "status": "error",
        "request_id": mw.current_request_id(),
    }
    if detail and settings.debug:
        body["detail"] = detail
    return body


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        _error_body(str(exc.detail)),
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        _error_body("Invalid request.", str(exc.errors())),
        status_code=422,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last resort. The traceback goes to the logs, never to the client -
    returning `traceback.format_exc()` in the response body leaks file
    paths and internal structure to anyone who can trigger an error.
    """
    logger.error(
        "Unhandled error on %s %s [%s]:\n%s",
        request.method, request.url.path, mw.current_request_id(),
        traceback.format_exc(),
    )
    return JSONResponse(
        _error_body("Internal server error.", str(exc)),
        status_code=500,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Deep health check reporting what is actually wired up."""
    from backend.services.speech_to_text import engine_ready, model_state  # type: ignore

    state = model_state()
    stt_ready = engine_ready()

    ml_status: dict = {"any_enabled": settings.any_ml_enabled}
    if settings.any_ml_enabled:
        # Importing the registry pulls in the ML package, so only touch it
        # when something is actually turned on.
        from backend.services.ml.registry import registry  # type: ignore
        ml_status.update(registry.status())

    return {
        "status": "ok" if stt_ready else "degraded",
        "version": APP_VERSION,
        "stt_engine": settings.resolved_stt_engine,
        "stt_ready": stt_ready,
        "local_model_loaded": state["loaded"],
        "llm_enabled": settings.llm_enabled,
        "active_sessions": session_store.active_count(),
        "config": settings.describe(),
        "ml": ml_status,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])


# ---------------------------------------------------------------------------
# Frontend (mounted last so it never shadows /api)
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)

    @app.get("/")
    async def no_frontend():
        return {"message": "SpeakTwin API running. Frontend not found."}


if __name__ == "__main__":
    import uvicorn  # type: ignore

    logger.info("Starting SpeakTwin on %s:%d", settings.host, settings.port)
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
