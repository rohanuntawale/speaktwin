"""
SpeakTwin - Confidence Score Calculator
========================================
Combines multiple speech metrics into a single 0-100 confidence score.

The score is a weighted average of normalised sub-scores:
  * WPM score          (25%)  - how close to optimal speaking speed
  * Pitch variation    (25%)  - vocal expressiveness
  * Energy score       (20%)  - appropriate loudness
  * Filler penalty     (30%)  - penalises excessive filler words

Loudness is scored on the dBFS scale with piecewise-linear ramps rather
than the old three-step RMS ladder, so a speaker drifting quiet sees the
score slide instead of dropping off a cliff.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from backend.utils.helpers import (  # type: ignore
    get_logger,
    CONFIDENCE_WEIGHTS,
    WPM_OPTIMAL_LOW,
    WPM_OPTIMAL_HIGH,
    PITCH_VARIATION_LOW,
    PITCH_VARIATION_GOOD,
    SILENCE_DBFS,
    ENERGY_LOW_DBFS,
    ENERGY_HIGH_DBFS,
    FILLER_RATE_HIGH,
    rms_to_dbfs,
)

logger = get_logger(__name__)


def _load_trained_weights() -> Dict[str, float]:
    """Load the small, checked-in fluency scorer trained on SpeechOcean762.

    Keep a safe fallback so a source checkout without the optional artifact
    still serves acoustic feedback.  The ASR model remains faster-whisper;
    these weights are the trained voice-confidence model used after ASR.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(project_root, "models", "voice", "confidence_weights_fluency.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            weights = json.load(handle).get("weights", {})
        loaded = {
            "wpm": float(weights["wpm"]),
            "pitch_variation": float(weights["pitch_variation"]),
            "energy": float(weights["energy"]),
            "filler_penalty": float(weights["filler_usage"]),
        }
        if all(value >= 0 for value in loaded.values()) and sum(loaded.values()) > 0:
            total = sum(loaded.values())
            return {key: value / total for key, value in loaded.items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Trained voice scorer unavailable; using fallback weights: %s", exc)
    return dict(CONFIDENCE_WEIGHTS)


TRAINED_CONFIDENCE_WEIGHTS = _load_trained_weights()

WPM_ZERO_SCORE_DEVIATION = 60.0   # WPM this far outside the band scores 0
TOO_LOUD_SPAN_DB = 6.0            # dB above the loud threshold to reach the floor
TOO_LOUD_FLOOR = 0.4


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def _wpm_score(wpm: float) -> float:
    """Optimal band scores 1.0; the score falls off linearly outside it."""
    if wpm <= 0:
        return 0.0
    if WPM_OPTIMAL_LOW <= wpm <= WPM_OPTIMAL_HIGH:
        return 1.0

    deviation = (
        WPM_OPTIMAL_LOW - wpm if wpm < WPM_OPTIMAL_LOW else wpm - WPM_OPTIMAL_HIGH
    )
    return _clamp(1.0 - (deviation / WPM_ZERO_SCORE_DEVIATION))


def _pitch_variation_score(pitch_std: float) -> float:
    """
    Score vocal expressiveness.

    Below PITCH_VARIATION_LOW reads as monotone; at or above
    PITCH_VARIATION_GOOD reads as expressive.
    """
    if pitch_std <= 0:
        return 0.0
    if pitch_std >= PITCH_VARIATION_GOOD:
        return 1.0
    if pitch_std <= PITCH_VARIATION_LOW:
        return _clamp(pitch_std / PITCH_VARIATION_LOW * 0.4)
    span = PITCH_VARIATION_GOOD - PITCH_VARIATION_LOW
    return _clamp(0.4 + 0.6 * (pitch_std - PITCH_VARIATION_LOW) / span)


def _energy_score(energy_db: float) -> float:
    """
    Score loudness on the dBFS scale.

      < SILENCE                 -> 0.0
      SILENCE .. LOW            -> ramps 0.0 -> 1.0
      LOW .. HIGH               -> 1.0 (the comfortable band)
      HIGH .. HIGH + 6 dB       -> ramps 1.0 -> 0.4
      beyond that               -> 0.4 (loud, but still speech)
    """
    if energy_db < SILENCE_DBFS:
        return 0.0
    if energy_db < ENERGY_LOW_DBFS:
        span = ENERGY_LOW_DBFS - SILENCE_DBFS
        return _clamp((energy_db - SILENCE_DBFS) / span) if span > 0 else 0.0
    if energy_db <= ENERGY_HIGH_DBFS:
        return 1.0
    over = energy_db - ENERGY_HIGH_DBFS
    return _clamp(1.0 - (1.0 - TOO_LOUD_FLOOR) * min(1.0, over / TOO_LOUD_SPAN_DB),
                  TOO_LOUD_FLOOR, 1.0)


def _filler_score(filler_rate: float) -> float:
    """0 fillers scores 1.0; twice the 'high' rate scores 0."""
    if filler_rate <= 0:
        return 1.0
    ceiling = FILLER_RATE_HIGH * 2
    if filler_rate >= ceiling:
        return 0.0
    return _clamp(1.0 - (filler_rate / ceiling))


def calculate_confidence(
    wpm: float,
    pitch_std: float,
    filler_rate: float,
    energy_db: float | None = None,
    energy: float | None = None,
) -> Dict[str, Any]:
    """
    Calculate the overall confidence score (0-100).

    Pass `energy_db` (dBFS). `energy` is accepted as a linear RMS fallback
    and converted, so older callers keep working.

    Returns
    -------
    dict with keys:
        score      : int  - final score 0-100
        breakdown  : dict - individual sub-scores (0-100 each)
    """
    if energy_db is None:
        energy_db = rms_to_dbfs(energy if energy is not None else 0.0)

    weights = TRAINED_CONFIDENCE_WEIGHTS
    sub = {
        "wpm": _wpm_score(wpm),
        "pitch_variation": _pitch_variation_score(pitch_std),
        "energy": _energy_score(energy_db),
        "filler_usage": _filler_score(filler_rate),
    }

    weighted = (
        sub["wpm"] * weights["wpm"]
        + sub["pitch_variation"] * weights["pitch_variation"]
        + sub["energy"] * weights["energy"]
        + sub["filler_usage"] * weights["filler_penalty"]
    )

    return {
        "score": int(round(_clamp(weighted) * 100)),
        "breakdown": {k: int(round(v * 100)) for k, v in sub.items()},
    }
