"""
SpeakTwin - Posture Coaching
=============================
Turns posture geometry and movement into a 0-100 score and messages a
speaker can act on, then fuses it with the voice score into one presence
number.

Two rules shape the copy here:

  * **Say what to do, not what is wrong.** "Level your shoulders" beats
    "shoulder asymmetry detected". The speaker is mid-practice and has
    about a second of attention to spare.
  * **Never coach on what was not seen.** If the hips were out of frame
    the torso lean is unknown, and reporting a confident 0 would be a
    lie. Unknown dimensions are dropped from the score, not zeroed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.utils.helpers import (  # type: ignore
    HEAD_SCALE_FIX,
    HEAD_TILT_NOTICEABLE,
    HEAD_TILT_PRONOUNCED,
    OPENNESS_CLOSED,
    OPENNESS_OPEN,
    POSTURE_WEIGHTS,
    PRESENCE_WEIGHTS,
    SHOULDER_TILT_NOTICEABLE,
    SHOULDER_TILT_PRONOUNCED,
    SWAY_RESTLESS,
    SWAY_STEADY,
    TORSO_LEAN_NOTICEABLE,
    TORSO_LEAN_PRONOUNCED,
    get_logger,
)

logger = get_logger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _decay(value: Optional[float], good: float, bad: float) -> Optional[float]:
    """
    1.0 at or below `good`, falling linearly to 0.0 at `bad`.

    None in, None out - an unmeasured dimension must not be scored.
    """
    if value is None:
        return None
    magnitude = abs(value)
    if magnitude <= good:
        return 1.0
    if magnitude >= bad:
        return 0.0
    return _clamp01(1.0 - (magnitude - good) / (bad - good))


def score_posture(pose: Dict[str, Any], movement: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine posture and movement into 0-100 with a per-dimension breakdown.

    Weights are renormalised over whatever was actually visible, so a
    speaker framed from the chest up is scored fairly on the dimensions
    the camera could see.
    """
    if not pose.get("detected"):
        return {"score": None, "breakdown": {}, "measured": []}

    # Alignment: shoulders level, head upright, torso vertical.
    alignment_parts = [
        _decay(pose.get("shoulder_tilt"), SHOULDER_TILT_NOTICEABLE, SHOULDER_TILT_PRONOUNCED * 1.6),
        _decay(pose.get("head_tilt"), HEAD_TILT_NOTICEABLE, HEAD_TILT_PRONOUNCED * 1.6),
        _decay(pose.get("torso_lean"), TORSO_LEAN_NOTICEABLE, TORSO_LEAN_PRONOUNCED * 1.6),
    ]
    measured_alignment = [p for p in alignment_parts if p is not None]
    alignment = sum(measured_alignment) / len(measured_alignment) if measured_alignment else None

    # Forward head: judged only against the speaker's calibrated baseline.
    # No baseline (no session, or first batch) → dimension excluded, not
    # zeroed — the absolute z-depth threshold this replaces nagged
    # correctly-seated people because of webcam perspective bias.
    deviation = pose.get("head_deviation")
    if deviation is None:
        head = None
    else:
        span = HEAD_SCALE_FIX - 1.0
        head = _clamp01(1.0 - (deviation - 1.0) / span) if span > 0 else None

    openness_value = pose.get("openness")
    if openness_value is None:
        openness = None
    elif openness_value >= OPENNESS_OPEN:
        openness = 1.0
    elif openness_value <= OPENNESS_CLOSED * 0.75:
        openness = 0.0
    else:
        span = OPENNESS_OPEN - OPENNESS_CLOSED * 0.75
        openness = _clamp01((openness_value - OPENNESS_CLOSED * 0.75) / span)

    steadiness = (
        _decay(movement.get("sway"), SWAY_STEADY, SWAY_RESTLESS * 1.8)
        if movement.get("detected") else None
    )

    dimensions = {
        "alignment": alignment,
        "head": head,
        "openness": openness,
        "steadiness": steadiness,
    }
    measured = {k: v for k, v in dimensions.items() if v is not None}
    if not measured:
        return {"score": None, "breakdown": {}, "measured": []}

    weight_total = sum(POSTURE_WEIGHTS[k] for k in measured)
    score = sum(v * POSTURE_WEIGHTS[k] for k, v in measured.items()) / weight_total

    return {
        "score": int(round(_clamp01(score) * 100)),
        "breakdown": {k: int(round(v * 100)) for k, v in measured.items()},
        "measured": sorted(measured),
    }


def generate_posture_feedback(pose: Dict[str, Any], movement: Dict[str, Any],
                              pose_bands: Dict[str, str],
                              movement_bands: Dict[str, str]) -> List[Dict[str, str]]:
    """Coaching messages, worst-first so the top one is worth acting on."""
    if not pose.get("detected"):
        return [{
            "text": "Step into frame so your head and shoulders are visible.",
            "type": "info",
            "category": "framing",
        }]

    messages: List[Dict[str, str]] = []
    add = lambda text, type_, category: messages.append(
        {"text": text, "type": type_, "category": category}
    )

    # ── Alignment ────────────────────────────────────────────────────
    if pose_bands.get("shoulder_tilt") == "pronounced":
        add("One shoulder is dropped. Level them out.", "warning", "shoulders")
    elif pose_bands.get("shoulder_tilt") == "noticeable":
        add("Shoulders are slightly uneven.", "info", "shoulders")
    elif pose_bands.get("shoulder_tilt") == "good":
        add("Shoulders are level.", "success", "shoulders")

    if pose_bands.get("head_tilt") == "pronounced":
        add("Your head is tilted. Bring it upright.", "warning", "head")
    elif pose_bands.get("head_tilt") == "noticeable":
        add("Slight head tilt.", "info", "head")

    if pose_bands.get("torso_lean") == "pronounced":
        add("You're leaning to one side. Centre your weight.", "warning", "torso")
    elif pose_bands.get("torso_lean") == "noticeable":
        add("Small lean off centre.", "info", "torso")

    # ── Forward head ─────────────────────────────────────────────────
    if pose_bands.get("forward_head") == "pronounced":
        add("Head is pushed forward. Draw your chin back over your shoulders.",
            "warning", "neck")
    elif pose_bands.get("forward_head") == "noticeable":
        add("Head drifting toward the screen.", "info", "neck")

    # ── Hand at the face ─────────────────────────────────────────────
    if pose_bands.get("hand_on_face") == "pronounced":
        add("Your hand is covering your face. Bring it down so your words carry.",
            "warning", "hands")
    elif pose_bands.get("hand_on_face") == "noticeable":
        add("Hand keeps drifting to your face.", "warning", "hands")

    # ── Openness ─────────────────────────────────────────────────────
    if pose_bands.get("openness") == "pronounced":
        add("You're hunched. Roll your shoulders back and lift your chest.",
            "warning", "openness")
    elif pose_bands.get("openness") == "noticeable":
        add("Open your chest a little.", "info", "openness")
    elif pose_bands.get("openness") == "good":
        add("Open, upright posture.", "success", "openness")

    # ── Movement ─────────────────────────────────────────────────────
    if movement_bands.get("sway") == "restless":
        add("You're swaying. Plant your feet and let your hands move instead.",
            "warning", "sway")
    elif movement_bands.get("sway") == "steady":
        add("Nice and steady.", "success", "sway")

    if movement_bands.get("gestures") == "too_still":
        add("Hands are static. Let them move with what you're saying.",
            "warning", "gestures")
    elif movement_bands.get("gestures") == "too_busy":
        add("Gestures are constant. Let some land and settle.", "warning", "gestures")
    elif movement_bands.get("gestures") == "good":
        add("Gestures look natural.", "success", "gestures")

    if movement_bands.get("fidget") == "high":
        add("A lot of small fidgeting. Rest your hands between gestures.",
            "warning", "fidget")

    # ── Hands in shot ────────────────────────────────────────────────
    hands = pose.get("hands_visible")
    if hands is not None and hands < 0.5:
        add("Your hands are out of frame. Bring them up where they can be seen.",
            "info", "hands")

    return messages


def posture_status(score: Optional[int], messages: List[Dict[str, str]]) -> str:
    if score is None:
        return "not_detected"
    warnings = sum(1 for m in messages if m["type"] == "warning")
    if warnings == 0:
        return "excellent"
    if warnings == 1:
        return "good"
    if warnings <= 3:
        return "needs_improvement"
    return "poor"


def presence_score(voice: Optional[int], body: Optional[int]) -> Optional[int]:
    """
    Fuse voice and body into one number.

    Whichever half is missing, the other stands alone rather than being
    halved — a speaker with no camera on should not appear to score 50.
    """
    if voice is None and body is None:
        return None
    if body is None:
        return voice
    if voice is None:
        return body

    weights = PRESENCE_WEIGHTS
    combined = voice * weights["voice"] + body * weights["body"]
    return int(round(combined))
