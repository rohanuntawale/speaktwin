"""
SpeakTwin - Speech Emotion Recognition
======================================
A wav2vec2 classifier reads affect straight from the waveform: tone,
tension, and energy that the transcript cannot carry.

For a speaking coach the useful signal is not the emotion label itself but
its bearing on delivery - a nervous or tense read is exactly what a speaker
wants flagged, and it correlates with things listeners notice long before
they notice word choice.

Runs once per chunk and is cheap enough for the 2.5s budget on CPU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np  # type: ignore

from backend.services.ml.registry import registry  # type: ignore
from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import SAMPLE_RATE, get_logger  # type: ignore

logger = get_logger(__name__)

MODEL_KEY = "emotion"
MODEL_SAMPLE_RATE = 16_000

# Label vocabularies differ sharply between checkpoints: the SUPERB/IEMOCAP
# models emit four-class abbreviations ("ang", "hap", "neu", "sad") while
# RAVDESS-trained ones emit full words. Both spellings are listed, because
# matching only the long form silently scores a furious delivery as calm.
TENSE_EMOTIONS = {
    "angry", "anger", "ang",
    "fear", "fearful", "fea",
    "sad", "sadness", "sadn",
    "disgust", "dis", "frustrated", "frustration",
}
POSITIVE_EMOTIONS = {
    "happy", "happiness", "hap", "joy",
    "excited", "excitement", "exc",
    "surprised", "surprise", "sur",
}
NEUTRAL_EMOTIONS = {"neutral", "neu", "calm"}


def _load():
    from transformers import pipeline  # type: ignore

    settings = get_settings()
    device = registry.device()
    # transformers wants -1 for CPU, or a device index.
    device_arg = -1 if device == "cpu" else 0

    return pipeline(
        task="audio-classification",
        model=settings.ml_emotion_model,
        device=device_arg,
        top_k=None,
    )


registry.register(
    MODEL_KEY,
    "Speech emotion recognition",
    "requirements-ml.txt",
    _load,
)


def is_enabled() -> bool:
    return get_settings().ml_emotion_enabled


def analyze(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[Dict[str, Any]]:
    """
    Classify the emotional tone of the chunk.

    Returns the top label, the full score distribution, and a derived
    `tension` value in 0-1 - or None when unavailable.
    """
    if not is_enabled():
        return None

    classifier = registry.get(MODEL_KEY)
    if classifier is None:
        return None

    if sr != MODEL_SAMPLE_RATE:
        logger.debug("Emotion model expects %d Hz, got %d", MODEL_SAMPLE_RATE, sr)
        return None

    # Very short chunks give the classifier nothing to work with.
    if len(audio) < MODEL_SAMPLE_RATE:
        return None

    try:
        with registry.infer_lock(MODEL_KEY):
            predictions = classifier(
                {"raw": np.ascontiguousarray(audio, dtype=np.float32),
                 "sampling_rate": MODEL_SAMPLE_RATE}
            )
    except Exception as exc:
        logger.warning("Emotion inference failed: %s", exc)
        return None

    if not predictions:
        return None

    scores = {
        str(p["label"]).lower(): round(float(p["score"]), 4)
        for p in predictions
    }
    top = max(scores.items(), key=lambda kv: kv[1])

    tension = sum(v for k, v in scores.items() if k in TENSE_EMOTIONS)
    positivity = sum(v for k, v in scores.items() if k in POSITIVE_EMOTIONS)

    # A label the checkpoint emits that we recognise in none of the three
    # buckets means the derived numbers are meaningless rather than zero.
    # Say so instead of quietly reporting a calm 0.0.
    known = TENSE_EMOTIONS | POSITIVE_EMOTIONS | NEUTRAL_EMOTIONS
    unmapped = sorted(k for k in scores if k not in known)
    if unmapped:
        logger.warning(
            "Emotion model emitted unrecognised labels %s - tension and "
            "positivity may be understated. Add them to the sets in %s.",
            unmapped, __name__,
        )

    return {
        "label": top[0],
        "confidence": top[1],
        "scores": scores,
        "tension": round(min(1.0, tension), 3),
        "positivity": round(min(1.0, positivity), 3),
        "unmapped_labels": unmapped,
        "engine": "wav2vec2",
    }
