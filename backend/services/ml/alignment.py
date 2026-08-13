"""
SpeakTwin - Word-Level Timestamps
==================================
Whisper returns segment-level timing at best, and the local path returned
nothing at all. Without per-word times, "pause ratio" is just a count of
quiet frames - it cannot say *where* the hesitation fell, which is the part
a speaker can act on.

Word timestamps unlock:
  * pauses located between specific words, not just totalled
  * speech rate measured over speaking time rather than wall-clock
  * transcript highlighting synchronised to playback
  * training labels for the disfluency model without hand-annotation

Two backends, tried in order:
  * **WhisperX** - forced alignment with a phoneme model. Most accurate,
    heaviest.
  * **whisper-timestamped** - dynamic-time-warping over attention weights.
    Lighter, no extra alignment model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)

MODEL_KEY = "alignment"
MODEL_SAMPLE_RATE = 16_000


def _load():
    """Load whichever alignment backend is configured and installed."""
    settings = get_settings()
    backend = settings.ml_alignment_backend.lower()

    if backend in ("auto", "whisperx"):
        try:
            import whisperx  # type: ignore

            model, metadata = whisperx.load_align_model(
                language_code="en", device=registry.device()
            )
            return {"backend": "whisperx", "module": whisperx,
                    "model": model, "metadata": metadata}
        except ImportError:
            if backend == "whisperx":
                raise
            logger.info("whisperx not installed, trying whisper-timestamped")

    import whisper_timestamped  # type: ignore

    return {
        "backend": "whisper_timestamped",
        "module": whisper_timestamped,
        "model": whisper_timestamped.load_model(
            settings.ml_alignment_whisper_model, device=registry.device()
        ),
    }


registry.register(
    MODEL_KEY,
    "Word-level timestamp alignment",
    "requirements-ml.txt",
    _load,
)


def is_enabled() -> bool:
    return get_settings().ml_alignment_enabled


def align(audio: np.ndarray, transcript: str,
          sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Produce word-level timings for an already-transcribed chunk.

    Returns the word list plus pause statistics derived from the gaps
    between words, or None when unavailable.
    """
    if not is_enabled() or not transcript.strip():
        return None

    bundle = registry.get(MODEL_KEY)
    if bundle is None or sr != MODEL_SAMPLE_RATE:
        return None

    audio = np.ascontiguousarray(audio, dtype=np.float32)

    try:
        with registry.infer_lock(MODEL_KEY):
            if bundle["backend"] == "whisperx":
                words = _align_whisperx(bundle, audio, transcript, sr)
            else:
                words = _align_whisper_timestamped(bundle, audio)
    except Exception as exc:
        logger.warning("Word alignment failed: %s", exc)
        return None

    if not words:
        return None

    return _summarise(words, duration=len(audio) / float(sr),
                      backend=bundle["backend"])


def _align_whisperx(bundle, audio: np.ndarray, transcript: str,
                    sr: int) -> List[Dict[str, Any]]:
    duration = len(audio) / float(sr)
    segments = [{"start": 0.0, "end": duration, "text": transcript}]

    result = bundle["module"].align(
        segments, bundle["model"], bundle["metadata"],
        audio, registry.device(), return_char_alignments=False,
    )

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if word.get("start") is None:
                continue
            words.append({
                "word": str(word.get("word", "")).strip(),
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "score": round(float(word.get("score", 0.0)), 3),
            })
    return words


def _align_whisper_timestamped(bundle, audio: np.ndarray) -> List[Dict[str, Any]]:
    result = bundle["module"].transcribe(
        bundle["model"], audio, language="en", vad=False,
    )

    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "word": str(word.get("text", "")).strip(),
                "start": round(float(word.get("start", 0.0)), 3),
                "end": round(float(word.get("end", 0.0)), 3),
                "score": round(float(word.get("confidence", 0.0)), 3),
            })
    return words


def _summarise(words: List[Dict[str, Any]], duration: float,
               backend: str) -> Dict[str, Any]:
    """Derive pause and pace statistics from word boundaries."""
    settings = get_settings()
    min_pause = settings.ml_alignment_min_pause

    pauses: List[Dict[str, float]] = []
    for previous, current in zip(words, words[1:]):
        gap = current["start"] - previous["end"]
        if gap >= min_pause:
            pauses.append({
                "after_word": previous["word"],
                "start": previous["end"],
                "duration": round(gap, 3),
            })

    speaking_seconds = sum(w["end"] - w["start"] for w in words)
    # Speech rate over time actually spent talking, not wall-clock - the
    # honest measure of how fast someone is speaking.
    articulation_rate = (
        round(len(words) / (speaking_seconds / 60.0), 1)
        if speaking_seconds > 0 else 0.0
    )

    return {
        "words": words,
        "word_count": len(words),
        "pauses": pauses,
        "pause_count": len(pauses),
        "longest_pause_sec": round(max((p["duration"] for p in pauses), default=0.0), 2),
        "speaking_seconds": round(speaking_seconds, 3),
        "articulation_rate_wpm": articulation_rate,
        "silence_ratio": round(1.0 - (speaking_seconds / duration), 3) if duration else 0.0,
        "engine": backend,
    }
