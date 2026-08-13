# ============================================================
# SpeakTwin - image WITH the deep learning layer
#
#   docker build -f Dockerfile.ml -t speaktwin:ml .
#   docker run -p 8000:8000 --env-file .env speaktwin:ml
#
# The base Dockerfile deliberately installs only requirements.txt, so
# every ML_*_ENABLED flag is inert there. This image adds torch and the
# model stack. It is roughly 3-4 GB versus ~1 GB for the base image, which
# is why they are separate rather than one image with an optional layer.
# ============================================================
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/models \
    TORCH_HOME=/models/torch \
    # Torch defaults to every core, which starves the request threadpool.
    ML_TORCH_THREADS=2

WORKDIR /app

# CPU-only torch first: the default wheel pulls the full CUDA stack and adds
# several gigabytes for nothing on a CPU host. Install it in its own layer so
# a change to the other requirements does not re-download it.
RUN pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Pre-download weights into an image layer so the first request does not pay
# for it. Each is guarded: a network failure at build time degrades to
# on-demand loading rather than failing the build.
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('tiny.en', device='cpu', compute_type='int8')" \
    || echo 'Whisper pre-download skipped'

RUN python -c "\
import torchcrepe, torch; \
torchcrepe.load.model(device='cpu', capacity='tiny')" \
    || echo 'CREPE pre-download skipped'

RUN python -c "\
import torch; \
torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)" \
    || echo 'Silero pre-download skipped'

RUN useradd --create-home --uid 10001 speaktwin \
    && mkdir -p /models \
    && chown -R speaktwin:speaktwin /app /models
USER speaktwin

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "\
import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=8).status==200 else 1)"

# Single worker: the session store and rate limiter are in-process, and each
# extra worker would also load its own copy of every enabled model.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
