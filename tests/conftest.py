"""
Shared pytest fixtures.

The environment is configured before any backend module is imported,
because `get_settings()` is cached for the life of the process.
"""

from __future__ import annotations

import io
import os

# --- Test environment (must precede backend imports) -----------------------
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LLM_ENABLED", "false")       # never call out to a provider
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "0")  # off, so tests can burst
os.environ.setdefault("MAX_UPLOAD_BYTES", "200000")  # 200 KB, small enough to trip
os.environ.setdefault("MAX_AUDIO_SECONDS", "30")
os.environ.setdefault("STT_ENGINE", "local")
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ.pop("API_KEY", None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Audio builders
# ---------------------------------------------------------------------------
def sine(frequency: float = 220.0, seconds: float = 2.5,
         amplitude: float = 0.2, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A steady tone - stands in for voiced speech."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def silence(seconds: float = 2.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def to_wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tone_wav() -> bytes:
    return to_wav_bytes(sine())


@pytest.fixture
def silence_wav() -> bytes:
    return to_wav_bytes(silence())


@pytest.fixture
def oversized_wav() -> bytes:
    """Larger than MAX_UPLOAD_BYTES, to exercise the 413 path."""
    return to_wav_bytes(sine(seconds=20.0))


@pytest.fixture
def stub_transcribe(monkeypatch):
    """
    Replace the STT call with a deterministic stub.

    Patches the name bound inside the route module, which is what the
    pipeline actually calls.
    """
    calls = []

    def _fake(audio, sr=SAMPLE_RATE, initial_prompt=None):
        calls.append({"samples": len(audio), "sr": sr, "prompt": initial_prompt})
        text = "So basically this is um a really innovative AI solution you know"
        return {"text": text, "word_count": len(text.split()), "segments": []}

    monkeypatch.setattr("backend.routes.analyze.transcribe", _fake)
    return calls


@pytest.fixture
def failing_transcribe(monkeypatch):
    """STT that always errors, to exercise the degraded path."""
    def _fake(audio, sr=SAMPLE_RATE, initial_prompt=None):
        return {"error": "engine exploded"}

    monkeypatch.setattr("backend.routes.analyze.transcribe", _fake)


@pytest.fixture
def client():
    """FastAPI test client with a clean session store per test."""
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.services.session_store import session_store

    session_store.clear()
    with TestClient(app) as test_client:
        yield test_client
    session_store.clear()
