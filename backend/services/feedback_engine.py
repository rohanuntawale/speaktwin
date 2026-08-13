"""
SpeakTwin - Feedback Engine
============================
Generates human-readable, real-time coaching messages from the acoustic
and linguistic metrics.

Threshold-based rules with an extensible shape that can later incorporate
ML-based classification.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.utils.helpers import (  # type: ignore
    get_logger,
    SILENCE_DBFS,
    ENERGY_LOW_DBFS,
    ENERGY_HIGH_DBFS,
    PITCH_LOW_THRESHOLD,
    PITCH_HIGH_THRESHOLD,
    PITCH_VARIATION_LOW,
    PITCH_VARIATION_GOOD,
    WPM_TOO_SLOW,
    WPM_TOO_FAST,
    WPM_OPTIMAL_LOW,
    WPM_OPTIMAL_HIGH,
    FILLER_RATE_HIGH,
    PAUSE_RATIO_HIGH,
    PAUSE_RATIO_NATURAL,
    rms_to_dbfs,
)

logger = get_logger(__name__)


def generate_feedback(
    mean_pitch: float,
    pitch_std: float,
    wpm: float,
    filler_rate: float,
    pause_ratio: float,
    energy_db: float | None = None,
    energy: float | None = None,
) -> Dict[str, Any]:
    """
    Produce a list of feedback messages and an overall status.

    Parameters
    ----------
    mean_pitch   : average pitch in Hz (0 when nothing voiced was found)
    pitch_std    : pitch standard deviation in Hz
    wpm          : words per minute
    filler_rate  : fillers per word
    pause_ratio  : fraction of silent frames
    energy_db    : loudness in dBFS (preferred)
    energy       : linear RMS fallback, converted when energy_db is omitted

    Returns
    -------
    dict with keys:
        messages : list[dict] - each has 'text', 'type', 'category'
        status   : str        - overall status label
    """
    if energy_db is None:
        energy_db = rms_to_dbfs(energy if energy is not None else 0.0)

    messages: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # 1. Loudness
    # ------------------------------------------------------------------
    if energy_db < SILENCE_DBFS:
        messages.append({
            "text": "Silence detected. Speak louder!",
            "type": "info",
            "category": "energy",
        })
    elif energy_db < ENERGY_LOW_DBFS:
        messages.append({
            "text": "Volume low. Project more!",
            "type": "warning",
            "category": "energy",
        })
    elif energy_db > ENERGY_HIGH_DBFS:
        messages.append({
            "text": "Volume high! Lower slightly.",
            "type": "warning",
            "category": "energy",
        })
    else:
        messages.append({
            "text": "Volume is perfect.",
            "type": "success",
            "category": "energy",
        })

    # ------------------------------------------------------------------
    # 2. Pitch
    # ------------------------------------------------------------------
    if mean_pitch > 0:  # only when something voiced was actually detected
        if mean_pitch < PITCH_LOW_THRESHOLD:
            messages.append({
                "text": "Pitch low. Sounds monotone.",
                "type": "warning",
                "category": "pitch",
            })
        elif mean_pitch > PITCH_HIGH_THRESHOLD:
            messages.append({
                "text": "Pitch high. Relax your voice.",
                "type": "warning",
                "category": "pitch",
            })
        else:
            messages.append({
                "text": "Comfortable vocal pitch.",
                "type": "success",
                "category": "pitch",
            })

        if pitch_std < PITCH_VARIATION_LOW:
            messages.append({
                "text": "Add variety! Tone is monotone.",
                "type": "warning",
                "category": "pitch_variation",
            })
        elif pitch_std >= PITCH_VARIATION_GOOD:
            messages.append({
                "text": "Great expression! Dynamic voice.",
                "type": "success",
                "category": "pitch_variation",
            })

    # ------------------------------------------------------------------
    # 3. Speaking speed
    # ------------------------------------------------------------------
    if wpm > 0:
        if wpm < WPM_TOO_SLOW:
            messages.append({
                "text": f"Too slow ({int(wpm)} WPM). Pick up pace!",
                "type": "warning",
                "category": "wpm",
            })
        elif wpm > WPM_TOO_FAST:
            messages.append({
                "text": f"Too fast ({int(wpm)} WPM)! Slow down.",
                "type": "warning",
                "category": "wpm",
            })
        elif WPM_OPTIMAL_LOW <= wpm <= WPM_OPTIMAL_HIGH:
            messages.append({
                "text": f"Excellent pace ({int(wpm)} WPM).",
                "type": "success",
                "category": "wpm",
            })
        else:
            messages.append({
                "text": f"Acceptable speed ({int(wpm)} WPM).",
                "type": "info",
                "category": "wpm",
            })

    # ------------------------------------------------------------------
    # 4. Filler words
    # ------------------------------------------------------------------
    if filler_rate > FILLER_RATE_HIGH:
        messages.append({
            "text": "Too many fillers! Use fewer 'ums'.",
            "type": "warning",
            "category": "fillers",
        })
    elif filler_rate > 0:
        messages.append({
            "text": "Good control of filler words.",
            "type": "info",
            "category": "fillers",
        })
    else:
        messages.append({
            "text": "No fillers! Perfect clarity.",
            "type": "success",
            "category": "fillers",
        })

    # ------------------------------------------------------------------
    # 5. Pauses
    # ------------------------------------------------------------------
    if pause_ratio > PAUSE_RATIO_HIGH:
        messages.append({
            "text": "Long pauses detected. Keep your speech flowing.",
            "type": "warning",
            "category": "pauses",
        })
    elif pause_ratio > PAUSE_RATIO_NATURAL:
        messages.append({
            "text": "Natural pacing with good pauses.",
            "type": "success",
            "category": "pauses",
        })

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    warning_count = sum(1 for m in messages if m["type"] == "warning")
    success_count = sum(1 for m in messages if m["type"] == "success")

    if energy_db < SILENCE_DBFS:
        status = "silent"
    elif warning_count == 0 and success_count > 0:
        status = "excellent"
    elif warning_count <= 1:
        status = "good"
    elif warning_count <= 3:
        status = "needs_improvement"
    else:
        status = "poor"

    return {"messages": messages, "status": status}
