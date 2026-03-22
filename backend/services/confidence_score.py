"""
SpeakTwin - Confidence Score Calculator
==========================================
Combines multiple speech metrics into a single 0-100 confidence score.

The score is a weighted average of normalised sub-scores:
  • WPM score          (25%)  – how close to optimal speaking speed
  • Pitch variation    (25%)  – vocal expressiveness
  • Energy score       (20%)  – appropriate volume
  • Filler penalty     (30%)  – penalises excessive filler words
"""

import numpy as np

from backend.utils.helpers import (
    get_logger,
    CONFIDENCE_WEIGHTS,
    WPM_OPTIMAL_LOW,
    WPM_OPTIMAL_HIGH,
    PITCH_VARIATION_LOW,
    PITCH_VARIATION_GOOD,
    ENERGY_SILENCE_THRESHOLD,
    ENERGY_LOW_THRESHOLD,
    ENERGY_HIGH_THRESHOLD,
    FILLER_RATE_HIGH,
)

logger = get_logger(__name__)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def _wpm_score(wpm: float) -> float:
    """
    Score WPM on a 0-1 scale.
    Optimal range [OPTIMAL_LOW, OPTIMAL_HIGH] → 1.0
    Linearly decreases outside that range.
    """
    if wpm <= 0:
        return 0.0
    mid = (WPM_OPTIMAL_LOW + WPM_OPTIMAL_HIGH) / 2
    half_range = (WPM_OPTIMAL_HIGH - WPM_OPTIMAL_LOW) / 2

    if WPM_OPTIMAL_LOW <= wpm <= WPM_OPTIMAL_HIGH:
        return 1.0

    # Distance from nearest optimal bound
    if wpm < WPM_OPTIMAL_LOW:
        deviation = WPM_OPTIMAL_LOW - wpm
    else:
        deviation = wpm - WPM_OPTIMAL_HIGH

    # Normalise deviation (60+ WPM off is score 0)
    return _clamp(1.0 - (deviation / 60.0))


def _pitch_variation_score(pitch_std: float) -> float:
    """
    Score pitch variation on a 0-1 scale.
    Below PITCH_VARIATION_LOW → monotone, penalised.
    Above PITCH_VARIATION_GOOD → expressive, rewarded.
    """
    if pitch_std <= 0:
        return 0.0
    if pitch_std >= PITCH_VARIATION_GOOD:
        return 1.0
    if pitch_std <= PITCH_VARIATION_LOW:
        return _clamp(pitch_std / PITCH_VARIATION_LOW * 0.4)
    # Linear interpolation between LOW and GOOD
    span = PITCH_VARIATION_GOOD - PITCH_VARIATION_LOW
    return _clamp(0.4 + 0.6 * (pitch_std - PITCH_VARIATION_LOW) / span)


def _energy_score(energy: float) -> float:
    """
    Score energy (volume) on a 0-1 scale.
    Too quiet or too loud → penalised.
    Good range → 1.0
    """
    if energy < ENERGY_SILENCE_THRESHOLD:
        return 0.0
    if energy < ENERGY_LOW_THRESHOLD:
        return 0.3
    if energy > ENERGY_HIGH_THRESHOLD:
        return 0.5
    # In the ideal range
    return 1.0


def _filler_score(filler_rate: float) -> float:
    """
    Score filler usage on a 0-1 scale.
    0 fillers → 1.0 (perfect)
    Above FILLER_RATE_HIGH → drops toward 0
    """
    if filler_rate <= 0:
        return 1.0
    if filler_rate >= FILLER_RATE_HIGH * 2:
        return 0.0
    return _clamp(1.0 - (filler_rate / (FILLER_RATE_HIGH * 2)))


def calculate_confidence(
    wpm: float,
    pitch_std: float,
    energy: float,
    filler_rate: float,
) -> dict:
    """
    Calculate the overall confidence score (0-100).

    Returns
    -------
    dict with keys:
        score       : int   – final score 0-100
        breakdown   : dict  – individual sub-scores (0-100 each)
    """
    w = CONFIDENCE_WEIGHTS
    sub = {
        "wpm": _wpm_score(wpm),
        "pitch_variation": _pitch_variation_score(pitch_std),
        "energy": _energy_score(energy),
        "filler_usage": _filler_score(filler_rate),
    }

    weighted = (
        sub["wpm"] * w["wpm"]
        + sub["pitch_variation"] * w["pitch_variation"]
        + sub["energy"] * w["energy"]
        + sub["filler_usage"] * w["filler_penalty"]
    )

    score = int(round(_clamp(weighted) * 100))

    breakdown = {k: int(round(v * 100)) for k, v in sub.items()}

    return {
        "score": score,
        "breakdown": breakdown,
    }
