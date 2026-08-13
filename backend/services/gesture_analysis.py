"""
SpeakTwin - Gesture & Movement
===============================
Where `pose_analysis` asks "what shape is this person in right now",
this module asks "how are they moving over time" - which is where the
difference between gesturing, fidgeting, and standing frozen lives.

Three distinct behaviours share one raw signal (wrist motion) and are
separated by scale and duration:

  * **gesture**   - purposeful travel, above GESTURE_MOTION_THRESHOLD
  * **fidget**    - small, fast, continuous movement that never resolves
  * **stillness** - almost no movement at all, which reads as rigid

All distances are in shoulder-widths, so nothing changes when the
speaker moves nearer to or further from the camera.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from backend.utils.helpers import (  # type: ignore
    FIDGET_MOTION_THRESHOLD,
    GESTURE_MOTION_THRESHOLD,
    GESTURE_RATE_HIGH,
    GESTURE_RATE_LOW,
    SWAY_RESTLESS,
    SWAY_STEADY,
    get_logger,
)

logger = get_logger(__name__)


def _distance(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.dist(a, b)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def analyse_movement(frames: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
    """
    Derive movement behaviour from the per-frame geometry series.

    `frames` are the usable frames produced by `pose_analysis.analyse_frames`.
    """
    empty = {
        "detected": False,
        "gesture_count": 0,
        "gesture_rate": 0.0,
        "gesture_amplitude": 0.0,
        "sway": 0.0,
        "fidget_ratio": 0.0,
        "stillness": 0.0,
    }
    if len(frames) < 2 or duration_sec <= 0:
        return empty

    # ── Wrist travel between consecutive frames ──────────────────────
    # Normalised by that frame's shoulder width so a speaker leaning in
    # toward the camera does not read as suddenly gesturing more.
    motions: List[float] = []
    for previous, current in zip(frames, frames[1:]):
        width = current.get("shoulder_width") or 0.0
        if width <= 1e-4:
            continue

        per_hand = []
        for index in (0, 1):
            a = (previous.get("wrists") or [None, None])[index]
            b = (current.get("wrists") or [None, None])[index]
            step = _distance(a, b)
            if step is not None:
                per_hand.append(step / width)

        if per_hand:
            motions.append(max(per_hand))   # the busier hand leads

    if not motions:
        return {**empty, "detected": True}

    # ── Gesture events ───────────────────────────────────────────────
    # A gesture is a *run* of motion above threshold, not a count of
    # frames: one sweeping arm movement is one gesture, not fifteen.
    gestures = 0
    amplitudes: List[float] = []
    run_peak = 0.0
    in_gesture = False

    for motion in motions:
        if motion >= GESTURE_MOTION_THRESHOLD:
            if not in_gesture:
                in_gesture = True
                gestures += 1
                run_peak = motion
            else:
                run_peak = max(run_peak, motion)
        elif in_gesture:
            in_gesture = False
            amplitudes.append(run_peak)
    if in_gesture:
        amplitudes.append(run_peak)

    minutes = duration_sec / 60.0
    gesture_rate = gestures / minutes if minutes > 0 else 0.0

    # ── Fidgeting ────────────────────────────────────────────────────
    # Small persistent motion that never rises to a real gesture.
    fidget_frames = sum(
        1 for m in motions
        if FIDGET_MOTION_THRESHOLD <= m < GESTURE_MOTION_THRESHOLD
    )
    fidget_ratio = fidget_frames / len(motions)

    # ── Body sway ────────────────────────────────────────────────────
    centres = [f["centre"] for f in frames if f.get("centre")]
    widths = [f.get("shoulder_width") or 0 for f in frames if f.get("shoulder_width")]
    reference_width = _mean(widths) if widths else 1.0

    if len(centres) >= 2 and reference_width > 1e-4:
        sway = (
            _stdev([c[0] for c in centres]) + _stdev([c[1] for c in centres])
        ) / reference_width
    else:
        sway = 0.0

    # ── Stillness ────────────────────────────────────────────────────
    # 1.0 means locked in place. Some stillness is poise; total stillness
    # reads as rigid, which is why this is reported rather than rewarded.
    still_frames = sum(1 for m in motions if m < FIDGET_MOTION_THRESHOLD)
    stillness = still_frames / len(motions)

    return {
        "detected": True,
        "gesture_count": gestures,
        "gesture_rate": round(gesture_rate, 1),
        "gesture_amplitude": round(_mean(amplitudes), 3),
        "sway": round(sway, 4),
        "fidget_ratio": round(fidget_ratio, 3),
        "stillness": round(stillness, 3),
        "motion_mean": round(_mean(motions), 4),
    }


def interpret(movement: Dict[str, Any]) -> Dict[str, str]:
    """Reduce movement numbers to coaching bands."""
    if not movement.get("detected"):
        return {}

    rate = movement.get("gesture_rate", 0.0)
    if rate < GESTURE_RATE_LOW:
        gesture_band = "too_still"
    elif rate > GESTURE_RATE_HIGH:
        gesture_band = "too_busy"
    else:
        gesture_band = "good"

    sway = movement.get("sway", 0.0)
    if sway >= SWAY_RESTLESS:
        sway_band = "restless"
    elif sway >= SWAY_STEADY:
        sway_band = "noticeable"
    else:
        sway_band = "steady"

    fidget = movement.get("fidget_ratio", 0.0)
    fidget_band = "high" if fidget > 0.45 else "some" if fidget > 0.22 else "low"

    return {"gestures": gesture_band, "sway": sway_band, "fidget": fidget_band}
