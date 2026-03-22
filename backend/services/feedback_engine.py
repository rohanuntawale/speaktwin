"""
SpeakTwin - Feedback Engine
==============================
Generates human-readable, real-time feedback messages based on
acoustic analysis and linguistic metrics.

Uses threshold-based rules with an extensible architecture that
can later incorporate ML-based classification.
"""

from backend.utils.helpers import ( # type: ignore
    get_logger,
    ENERGY_SILENCE_THRESHOLD,
    ENERGY_LOW_THRESHOLD,
    ENERGY_HIGH_THRESHOLD,
    PITCH_LOW_THRESHOLD,
    PITCH_HIGH_THRESHOLD,
    PITCH_VARIATION_LOW,
    PITCH_VARIATION_GOOD,
    WPM_TOO_SLOW,
    WPM_TOO_FAST,
    WPM_OPTIMAL_LOW,
    WPM_OPTIMAL_HIGH,
    FILLER_RATE_HIGH,
)

logger = get_logger(__name__)


def generate_feedback(
    energy: float,
    mean_pitch: float,
    pitch_std: float,
    wpm: float,
    filler_rate: float,
    pause_ratio: float,
) -> dict:
    """
    Produce a list of feedback messages and an overall status.

    Parameters
    ----------
    energy       : float – RMS energy
    mean_pitch   : float – average pitch in Hz
    pitch_std    : float – pitch standard deviation
    wpm          : float – words per minute
    filler_rate  : float – fillers per word
    pause_ratio  : float – fraction of silent frames

    Returns
    -------
    dict with keys:
        messages : list[dict]  – each has 'text', 'type' (info/warning/success)
        status   : str         – overall status label
    """
    messages: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Energy / Volume Feedback
    # ------------------------------------------------------------------
    if energy < ENERGY_SILENCE_THRESHOLD:
        messages.append({
            "text": "Silence detected. Speak louder!",
            "type": "info",
            "category": "energy",
        })
    elif energy < ENERGY_LOW_THRESHOLD:
        messages.append({
            "text": "Volume low. Project more!",
            "type": "warning",
            "category": "energy",
        })
    elif energy > ENERGY_HIGH_THRESHOLD:
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
    # 2. Pitch Feedback
    # ------------------------------------------------------------------
    if mean_pitch > 0:  # only if pitch was detected
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

        # Pitch variation
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
    # 3. Speaking Speed (WPM) Feedback
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
    # 4. Filler Words Feedback
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
    # 5. Pause / Silence Feedback
    # ------------------------------------------------------------------
    if pause_ratio > 0.6:
        messages.append({
            "text": "Long pauses detected. Keep your speech flowing.",
            "type": "warning",
            "category": "pauses",
        })
    elif pause_ratio > 0.35:
        messages.append({
            "text": "Natural pacing with good pauses.",
            "type": "success",
            "category": "pauses",
        })

    # ------------------------------------------------------------------
    # Overall Status
    # ------------------------------------------------------------------
    warning_count = sum(1 for m in messages if m["type"] == "warning")
    success_count = sum(1 for m in messages if m["type"] == "success")

    if energy < ENERGY_SILENCE_THRESHOLD:
        status = "silent"
    elif warning_count == 0 and success_count > 0:
        status = "excellent"
    elif warning_count <= 1:
        status = "good"
    elif warning_count <= 3:
        status = "needs_improvement"
    else:
        status = "poor"

    return {
        "messages": messages,
        "status": status,
    }
