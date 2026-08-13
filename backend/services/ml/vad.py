"""
SpeakTwin - Neural Voice Activity Detection (Silero VAD)
=========================================================
Silero VAD is a small, fast RNN that decides which samples contain speech.
It replaces the energy-threshold gate with something that actually
distinguishes speech from noise, which matters for two reasons:

  * an energy gate cannot tell a quiet speaker from a noisy room, so it
    either transcribes hiss or drops real speech
  * pause metrics computed from energy count background noise as speech

It also gives honest speech timing, so pause ratio and speech rate are
measured against time actually spent talking rather than wall-clock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)

MODEL_KEY = "silero_vad"
VAD_SAMPLE_RATE = 16_000


def _load():
    import torch  # type: ignore

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    model.to(registry.device())
    # utils = (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    return {"model": model, "get_speech_timestamps": utils[0]}


registry.register(
    MODEL_KEY,
    "Silero voice activity detection",
    "requirements-ml.txt",
    _load,
)


def is_enabled() -> bool:
    return get_settings().ml_vad_enabled


def detect_speech(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Locate speech regions in the chunk.

    Returns speech ratio, segment boundaries in seconds, the longest silent
    gap, and total speech duration - or None when unavailable.
    """
    if not is_enabled():
        return None

    bundle = registry.get(MODEL_KEY)
    if bundle is None:
        return None

    if sr != VAD_SAMPLE_RATE:
        logger.debug("Silero VAD expects %d Hz, got %d - skipping", VAD_SAMPLE_RATE, sr)
        return None

    settings = get_settings()

    try:
        import torch  # type: ignore

        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

        with registry.infer_lock(MODEL_KEY):
            with torch.no_grad():
                stamps = bundle["get_speech_timestamps"](
                    tensor,
                    bundle["model"],
                    sampling_rate=VAD_SAMPLE_RATE,
                    threshold=settings.ml_vad_threshold,
                    min_speech_duration_ms=120,
                    min_silence_duration_ms=180,
                )
    except Exception as exc:
        logger.warning("Silero VAD inference failed: %s", exc)
        return None

    duration = len(audio) / float(sr) if sr else 0.0
    if duration <= 0:
        return None

    segments: List[Dict[str, float]] = [
        {"start": round(s["start"] / VAD_SAMPLE_RATE, 3),
         "end": round(s["end"] / VAD_SAMPLE_RATE, 3)}
        for s in stamps
    ]

    speech_seconds = sum(s["end"] - s["start"] for s in segments)

    # Longest gap, including the head and tail of the chunk.
    longest_gap = 0.0
    cursor = 0.0
    for segment in segments:
        longest_gap = max(longest_gap, segment["start"] - cursor)
        cursor = segment["end"]
    longest_gap = max(longest_gap, duration - cursor)

    return {
        "has_speech": bool(segments),
        "speech_ratio": round(speech_seconds / duration, 3),
        "speech_seconds": round(speech_seconds, 3),
        "pause_ratio": round(1.0 - (speech_seconds / duration), 3),
        "longest_pause_sec": round(longest_gap, 2),
        "segments": segments,
        "engine": "silero",
    }
