# ============================================================
# SpeakTwin - AI Communication Mirror
# ============================================================
FROM python:3.11-slim

# libsndfile is what soundfile binds to for audio decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    # Cache Whisper weights inside the image layer, not the container's
    # writable layer, so a restart does not re-download them.
    HF_HOME=/models

WORKDIR /app

# Dependencies first so a source change does not invalidate the layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Pre-download the local Whisper model so the first request is not slow.
# Harmless if it fails - the app falls back to loading on demand.
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('tiny.en', device='cpu', compute_type='int8')" || \
    echo "Model pre-download skipped"

# Run unprivileged.
RUN useradd --create-home --uid 10001 speaktwin \
    && chown -R speaktwin:speaktwin /app /models
USER speaktwin

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "\
import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

# Single worker: the session store and rate limiter are in-process, so
# multiple workers would each keep their own copy. Scale with replicas
# plus a shared store if you need more throughput.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
