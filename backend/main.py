"""
SpeakTwin – AI Communication Mirror
======================================
FastAPI Application Entry Point

This is the main server file that:
  1. Initialises the FastAPI app with CORS middleware
  2. Mounts the frontend static files
  3. Registers API routes
  4. Pre-loads ML models on startup for fast inference
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import FileResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore

from backend.routes.analyze import router as analyze_router # type: ignore
from backend.utils.helpers import get_logger # type: ignore

logger = get_logger("speaktwin")

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


# ---------------------------------------------------------------------------
# Lifespan – model pre-loading on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load expensive ML models at startup."""
    logger.info("=" * 60)
    logger.info("  SpeakTwin – AI Communication Mirror")
    logger.info("  Starting up...")
    logger.info("=" * 60)

    # Pre-load the Whisper STT model (singleton)
    try:
        from backend.services.speech_to_text import _load_model # type: ignore
        _load_model()
        logger.info("STT model ready")
    except Exception as e:
        logger.warning("Could not pre-load STT model: %s", e)

    yield  # App runs here

    logger.info("SpeakTwin shutting down...")
    # Clean up audio capture if initialized (deployment safe)
    try:
        from backend.services.audio_capture import AudioCapture # type: ignore
        # Accessing the instance only if it was already created or is safely initializable
        if AudioCapture._instance:
            AudioCapture().stop()
    except Exception as e:
        logger.debug("Audio cleanup skipped: %s", e)


# ---------------------------------------------------------------------------
# App Initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SpeakTwin – AI Communication Mirror",
    description="Real-time speech analysis and coaching feedback",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow frontend (even from file:// or different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register Routes
# ---------------------------------------------------------------------------
app.include_router(analyze_router, prefix="/api", tags=["Analysis"])

# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
# Mount static assets (CSS, JS)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")

    @app.get("/")
    async def serve_index():
        """Serve the main frontend page."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)

    @app.get("/")
    async def no_frontend():
        return {"message": "SpeakTwin API running. Frontend not found."}

if __name__ == "__main__":
    import uvicorn # type: ignore
    # Pull config from env or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    
    logger.info(f"Starting SpeakTwin Server on {host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
