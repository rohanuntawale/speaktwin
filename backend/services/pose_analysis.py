"""
SpeakTwin - Posture Geometry
=============================
Turns MediaPipe pose landmarks into the handful of angles and ratios that
actually describe how a speaker is carrying themselves.

Pose detection runs in the browser; only landmarks reach the server. That
keeps video on the speaker's machine and makes a batch ~25 KB instead of
megabytes of frames.

One correctness detail that is easy to get wrong: MediaPipe normalises x
and y to 0-1 **independently**, against a frame that is almost never
square. Computing an angle from those raw values stretches it by the
aspect ratio - a level pair of shoulders in a 16:9 frame measures as
tilted. Every angle here is computed in aspect-corrected space.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.utils.helpers import (  # type: ignore
    FORWARD_HEAD_NOTICEABLE,
    FORWARD_HEAD_PRONOUNCED,
    HEAD_TILT_NOTICEABLE,
    HEAD_TILT_PRONOUNCED,
    OPENNESS_CLOSED,
    OPENNESS_OPEN,
    POSE_LEFT_EAR,
    POSE_LEFT_HIP,
    POSE_LEFT_SHOULDER,
    POSE_LEFT_WRIST,
    POSE_MIN_VISIBILITY,
    POSE_NOSE,
    POSE_RIGHT_EAR,
    POSE_RIGHT_HIP,
    POSE_RIGHT_SHOULDER,
    POSE_RIGHT_WRIST,
    SHOULDER_TILT_NOTICEABLE,
    SHOULDER_TILT_PRONOUNCED,
    TORSO_LEAN_NOTICEABLE,
    TORSO_LEAN_PRONOUNCED,
    get_logger,
)

logger = get_logger(__name__)

Point = Tuple[float, float]


class PoseFrame:
    """
    One frame of landmarks in aspect-corrected space.

    Landmarks arrive as dicts with x, y, z and visibility. Anything below
    the visibility floor is treated as absent rather than trusted.
    """

    def __init__(self, landmarks: Sequence[Dict[str, float]], aspect: float = 1.0):
        self.raw = landmarks
        self.aspect = aspect if aspect and aspect > 0 else 1.0

    def _visible(self, index: int) -> bool:
        if index >= len(self.raw):
            return False
        point = self.raw[index]
        return float(point.get("visibility", 1.0)) >= POSE_MIN_VISIBILITY

    def point(self, index: int) -> Optional[Point]:
        """Aspect-corrected (x, y). None when the landmark is not reliable."""
        if not self._visible(index):
            return None
        point = self.raw[index]
        # Multiplying x by the aspect ratio restores real proportions, so
        # angles measured here match what the camera actually saw.
        return (float(point["x"]) * self.aspect, float(point["y"]))

    def depth(self, index: int) -> Optional[float]:
        if not self._visible(index):
            return None
        return float(self.raw[index].get("z", 0.0))

    def midpoint(self, a: int, b: int) -> Optional[Point]:
        pa, pb = self.point(a), self.point(b)
        if pa is None or pb is None:
            return None
        return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

    @property
    def shoulder_width(self) -> Optional[float]:
        left, right = self.point(POSE_LEFT_SHOULDER), self.point(POSE_RIGHT_SHOULDER)
        if left is None or right is None:
            return None
        width = math.dist(left, right)
        return width if width > 1e-4 else None

    @property
    def usable(self) -> bool:
        """Both shoulders visible is the minimum for any of this to mean anything."""
        return self.shoulder_width is not None


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------
def line_angle_from_horizontal(a: Point, b: Point) -> float:
    """Signed angle of a-b against horizontal, in degrees (-90..90)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx)) % 180.0 - (
        180.0 if math.degrees(math.atan2(dy, dx)) % 180.0 > 90 else 0.0
    )


def tilt_degrees(a: Point, b: Point) -> float:
    """How far a-b departs from level, 0-90 degrees."""
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    if dx < 1e-9:
        return 90.0
    return math.degrees(math.atan(dy / dx))


def lean_degrees(lower: Point, upper: Point) -> float:
    """How far the lower-upper line departs from vertical, 0-90 degrees."""
    dx = abs(upper[0] - lower[0])
    dy = abs(upper[1] - lower[1])
    if dy < 1e-9:
        return 90.0
    return math.degrees(math.atan(dx / dy))


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------
def analyse_frame(frame: PoseFrame) -> Optional[Dict[str, Any]]:
    """
    Geometry for one frame. None when the speaker is not usably in shot.

    Every distance is divided by shoulder width, so results do not change
    when the speaker moves closer to or further from the camera.
    """
    if not frame.usable:
        return None

    width = frame.shoulder_width
    left_sh = frame.point(POSE_LEFT_SHOULDER)
    right_sh = frame.point(POSE_RIGHT_SHOULDER)
    shoulder_mid = frame.midpoint(POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER)

    result: Dict[str, Any] = {
        "shoulder_tilt": round(tilt_degrees(left_sh, right_sh), 2),
        "shoulder_width": round(width, 4),
    }

    # ── Head tilt ────────────────────────────────────────────────────
    left_ear, right_ear = frame.point(POSE_LEFT_EAR), frame.point(POSE_RIGHT_EAR)
    if left_ear and right_ear:
        result["head_tilt"] = round(tilt_degrees(left_ear, right_ear), 2)

    # ── Head turned away from the audience ───────────────────────────
    nose = frame.point(POSE_NOSE)
    if nose and shoulder_mid:
        result["head_offset"] = round((nose[0] - shoulder_mid[0]) / width, 3)

    # ── Torso lean and openness ──────────────────────────────────────
    hip_mid = frame.midpoint(POSE_LEFT_HIP, POSE_RIGHT_HIP)
    if hip_mid and shoulder_mid:
        result["torso_lean"] = round(lean_degrees(hip_mid, shoulder_mid), 2)

        torso_height = math.dist(hip_mid, shoulder_mid)
        if torso_height > 1e-4:
            # Wide shoulders over a tall torso reads as upright and open;
            # the ratio collapses when someone hunches or curls inward.
            result["openness"] = round(width / torso_height, 3)

    # ── Forward head carriage ────────────────────────────────────────
    # The ears sitting ahead of the shoulders in depth is the "reading a
    # screen" posture. MediaPipe's z is relative and noisy, so this is
    # reported as an indication rather than a measurement.
    ear_depths = [d for d in (frame.depth(POSE_LEFT_EAR), frame.depth(POSE_RIGHT_EAR))
                  if d is not None]
    shoulder_depths = [d for d in (frame.depth(POSE_LEFT_SHOULDER),
                                   frame.depth(POSE_RIGHT_SHOULDER)) if d is not None]
    if ear_depths and shoulder_depths:
        ear_z = sum(ear_depths) / len(ear_depths)
        shoulder_z = sum(shoulder_depths) / len(shoulder_depths)
        # Negative z is nearer the camera in MediaPipe's convention.
        result["forward_head"] = round((shoulder_z - ear_z) / width, 3)

    # ── Hands ────────────────────────────────────────────────────────
    wrists = [frame.point(POSE_LEFT_WRIST), frame.point(POSE_RIGHT_WRIST)]
    result["hands_visible"] = sum(1 for w in wrists if w is not None)

    if shoulder_mid and hip_mid:
        # Hands held between waist and shoulders read as natural gesturing
        # space; below the waist reads as hidden or in pockets.
        in_box = 0
        for wrist in wrists:
            if wrist is None:
                continue
            if shoulder_mid[1] <= wrist[1] <= hip_mid[1] + (hip_mid[1] - shoulder_mid[1]) * 0.35:
                in_box += 1
        result["hands_in_gesture_box"] = in_box

    result["centre"] = [round(shoulder_mid[0], 4), round(shoulder_mid[1], 4)]
    result["wrists"] = [
        [round(w[0], 4), round(w[1], 4)] if w else None for w in wrists
    ]

    return result


def analyse_frames(frames: List[PoseFrame]) -> Dict[str, Any]:
    """
    Average the per-frame geometry across a batch.

    Averaging before judging keeps a single bad frame - a hand across the
    face, a detection blip - from producing a spurious warning.
    """
    per_frame = [analyse_frame(f) for f in frames]
    usable = [m for m in per_frame if m is not None]

    if not usable:
        return {"usable_frames": 0, "total_frames": len(frames), "detected": False}

    def mean(key: str) -> Optional[float]:
        values = [m[key] for m in usable if key in m and m[key] is not None]
        return round(sum(values) / len(values), 3) if values else None

    summary: Dict[str, Any] = {
        "detected": True,
        "usable_frames": len(usable),
        "total_frames": len(frames),
        "shoulder_tilt": mean("shoulder_tilt"),
        "head_tilt": mean("head_tilt"),
        "head_offset": mean("head_offset"),
        "torso_lean": mean("torso_lean"),
        "openness": mean("openness"),
        "forward_head": mean("forward_head"),
        "hands_visible": mean("hands_visible"),
        "hands_in_gesture_box": mean("hands_in_gesture_box"),
    }
    summary["frames"] = usable  # gesture analysis needs the time series
    return summary


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
def _band(value: Optional[float], noticeable: float, pronounced: float) -> str:
    if value is None:
        return "unknown"
    magnitude = abs(value)
    if magnitude >= pronounced:
        return "pronounced"
    if magnitude >= noticeable:
        return "noticeable"
    return "good"


def interpret(summary: Dict[str, Any]) -> Dict[str, str]:
    """Reduce each raw measurement to good / noticeable / pronounced."""
    if not summary.get("detected"):
        return {}

    bands = {
        "shoulder_tilt": _band(summary.get("shoulder_tilt"),
                               SHOULDER_TILT_NOTICEABLE, SHOULDER_TILT_PRONOUNCED),
        "head_tilt": _band(summary.get("head_tilt"),
                           HEAD_TILT_NOTICEABLE, HEAD_TILT_PRONOUNCED),
        "torso_lean": _band(summary.get("torso_lean"),
                            TORSO_LEAN_NOTICEABLE, TORSO_LEAN_PRONOUNCED),
        "forward_head": _band(summary.get("forward_head"),
                              FORWARD_HEAD_NOTICEABLE, FORWARD_HEAD_PRONOUNCED),
    }

    openness = summary.get("openness")
    if openness is None:
        bands["openness"] = "unknown"
    elif openness >= OPENNESS_OPEN:
        bands["openness"] = "good"
    elif openness >= OPENNESS_CLOSED:
        bands["openness"] = "noticeable"
    else:
        bands["openness"] = "pronounced"

    return bands
