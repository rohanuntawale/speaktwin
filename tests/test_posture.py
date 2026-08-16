"""Posture geometry, gesture analysis, scoring, and the /api/pose endpoint."""

from __future__ import annotations

import math

import pytest

from backend.services.gesture_analysis import analyse_movement
from backend.services.gesture_analysis import interpret as interpret_movement
from backend.services.pose_analysis import (
    PoseFrame,
    analyse_frame,
    analyse_frames,
    interpret,
    lean_degrees,
    tilt_degrees,
)
from backend.services.posture_feedback import (
    generate_posture_feedback,
    presence_score,
    score_posture,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def lm(x, y, z=0.0, v=1.0):
    return {"x": x, "y": y, "z": z, "visibility": v}


def body(shoulder_dy=0.0, hip_dx=0.0, torso=0.25, wrist_x=0.36,
         ear_z=0.0, ear_dx=0.045, hand_at_mouth=False, visible=True):
    """
    A plausible upright speaker. `shoulder_dy` drops the right shoulder,
    `hip_dx` shifts the hips sideways to create a lean, `ear_dx` widens
    the apparent head (larger = head nearer the camera), `hand_at_mouth`
    raises the right hand to cover the mouth.
    """
    v = 1.0 if visible else 0.0
    points = [lm(0.50, 0.24, v=v)]                    # 0 nose
    points += [lm(0.50, 0.24, v=v) for _ in range(6)]  # 1-6 eyes
    points += [lm(0.50 - ear_dx, 0.235, ear_z, v),
               lm(0.50 + ear_dx, 0.235, ear_z, v)]     # 7,8 ears
    points += [lm(0.50, 0.28, v=v), lm(0.50, 0.28, v=v)]                # 9,10 mouth
    points += [lm(0.40, 0.40 - shoulder_dy, 0.0, v),
               lm(0.60, 0.40 + shoulder_dy, 0.0, v)]                    # 11,12 shoulders
    points += [lm(0.36, 0.55, v=v), lm(0.64, 0.55, v=v)]                # 13,14 elbows
    if hand_at_mouth:
        points += [lm(wrist_x, 0.66, v=v), lm(0.52, 0.31, v=v)]         # 15,16 wrists
        points += [lm(0.50, 0.70, v=v) for _ in range(3)]               # 17,18,19 (left)
        points += [lm(0.50, 0.285, v=v)]                                # 20 R index on mouth
        points += [lm(0.50, 0.70, v=v) for _ in range(2)]               # 21,22
    else:
        points += [lm(wrist_x, 0.66, v=v), lm(1 - wrist_x, 0.66, v=v)]  # 15,16 wrists
        points += [lm(0.50, 0.70, v=v) for _ in range(6)]               # 17-22 hands
    points += [lm(0.44 + hip_dx, 0.40 + torso, 0.0, v),
               lm(0.56 + hip_dx, 0.40 + torso, 0.0, v)]                 # 23,24 hips
    points += [lm(0.50, 0.95, v=v) for _ in range(8)]                   # 25-32 legs
    return points


def frames(count=20, **kwargs):
    return [PoseFrame(body(**kwargs), aspect=1.0) for _ in range(count)]


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
def test_level_line_has_no_tilt():
    assert tilt_degrees((0.0, 0.5), (1.0, 0.5)) == pytest.approx(0.0)


def test_tilt_is_measured_in_degrees():
    assert tilt_degrees((0.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)


def test_vertical_line_has_no_lean():
    assert lean_degrees((0.5, 1.0), (0.5, 0.0)) == pytest.approx(0.0)


def test_lean_is_measured_from_vertical():
    assert lean_degrees((0.0, 1.0), (1.0, 0.0)) == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Aspect-ratio correction
# ---------------------------------------------------------------------------
def test_aspect_ratio_changes_the_measured_angle():
    """
    MediaPipe normalises x and y independently, so the same landmarks in a
    16:9 frame describe a different real-world angle than in a square one.
    Ignoring that reports level shoulders as tilted.
    """
    square = analyse_frame(PoseFrame(body(shoulder_dy=0.03), aspect=1.0))
    wide = analyse_frame(PoseFrame(body(shoulder_dy=0.03), aspect=16 / 9))

    assert square["shoulder_tilt"] > wide["shoulder_tilt"]


def test_level_shoulders_read_level_at_any_aspect():
    for aspect in (1.0, 4 / 3, 16 / 9):
        result = analyse_frame(PoseFrame(body(), aspect=aspect))
        assert result["shoulder_tilt"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Per-frame geometry
# ---------------------------------------------------------------------------
def test_upright_body_reads_well():
    result = analyse_frame(PoseFrame(body(), aspect=1.0))
    assert result["shoulder_tilt"] == pytest.approx(0.0, abs=0.5)
    assert result["torso_lean"] == pytest.approx(0.0, abs=0.5)
    assert result["hands_visible"] == 2


def test_dropped_shoulder_is_detected():
    result = analyse_frame(PoseFrame(body(shoulder_dy=0.06), aspect=1.0))
    assert result["shoulder_tilt"] > 10


def test_sideways_hips_produce_a_lean():
    result = analyse_frame(PoseFrame(body(hip_dx=0.12), aspect=1.0))
    assert result["torso_lean"] > 10


def test_invisible_landmarks_yield_no_frame():
    assert analyse_frame(PoseFrame(body(visible=False), aspect=1.0)) is None


def test_openness_falls_as_the_torso_stretches():
    """Shoulder width over torso height — a longer torso reads as less open."""
    compact = analyse_frame(PoseFrame(body(torso=0.20), aspect=1.0))
    stretched = analyse_frame(PoseFrame(body(torso=0.40), aspect=1.0))
    assert compact["openness"] > stretched["openness"]


def test_metrics_are_scale_invariant():
    """Moving nearer the camera must not change the angles."""
    near = analyse_frame(PoseFrame(body(shoulder_dy=0.04), aspect=1.0))

    scaled = []
    for point in body(shoulder_dy=0.04):
        scaled.append(lm(0.5 + (point["x"] - 0.5) * 1.6,
                         0.5 + (point["y"] - 0.5) * 1.6,
                         point["z"], point["visibility"]))
    far = analyse_frame(PoseFrame(scaled, aspect=1.0))

    assert near["shoulder_tilt"] == pytest.approx(far["shoulder_tilt"], abs=0.5)


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------
def test_batch_reports_detection_and_counts():
    summary = analyse_frames(frames(12))
    assert summary["detected"] is True
    assert summary["usable_frames"] == 12
    assert summary["total_frames"] == 12


def test_batch_with_nobody_present():
    summary = analyse_frames(frames(8, visible=False))
    assert summary["detected"] is False
    assert summary["usable_frames"] == 0


def test_empty_batch_is_safe():
    summary = analyse_frames([])
    assert summary["detected"] is False


def test_interpretation_bands():
    good = interpret(analyse_frames(frames(6)))
    assert good["shoulder_tilt"] == "good"

    bad = interpret(analyse_frames(frames(6, shoulder_dy=0.09)))
    assert bad["shoulder_tilt"] == "pronounced"


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
def series(wrist_positions):
    """Build the per-frame series analyse_movement consumes."""
    return [
        {"centre": [0.5, 0.4], "shoulder_width": 0.2,
         "wrists": [[x, 0.66], [1 - x, 0.66]]}
        for x in wrist_positions
    ]


def test_still_hands_produce_no_gestures():
    movement = analyse_movement(series([0.36] * 20), duration_sec=2.0)
    assert movement["gesture_count"] == 0
    assert movement["stillness"] == 1.0


def test_one_sweep_counts_as_one_gesture():
    """A single arm movement is one gesture, not one per frame."""
    positions = [0.36] * 5 + [0.30, 0.24, 0.18] + [0.16] * 8
    movement = analyse_movement(series(positions), duration_sec=2.0)
    assert movement["gesture_count"] == 1


def test_two_separated_sweeps_count_twice():
    positions = [0.36] * 4 + [0.24, 0.16] + [0.16] * 6 + [0.26, 0.36] + [0.36] * 4
    movement = analyse_movement(series(positions), duration_sec=2.0)
    assert movement["gesture_count"] == 2


def test_gesture_rate_scales_with_duration():
    positions = [0.36] * 4 + [0.24, 0.16] + [0.16] * 8
    fast = analyse_movement(series(positions), duration_sec=1.0)
    slow = analyse_movement(series(positions), duration_sec=4.0)
    assert fast["gesture_rate"] > slow["gesture_rate"]


def test_sway_detects_a_moving_torso():
    steady = [{"centre": [0.5, 0.4], "shoulder_width": 0.2,
               "wrists": [[0.36, 0.66], [0.64, 0.66]]} for _ in range(20)]
    swaying = [{"centre": [0.5 + 0.03 * math.sin(i), 0.4], "shoulder_width": 0.2,
                "wrists": [[0.36, 0.66], [0.64, 0.66]]} for i in range(20)]

    assert analyse_movement(swaying, 2.0)["sway"] > analyse_movement(steady, 2.0)["sway"]


def test_movement_needs_at_least_two_frames():
    assert analyse_movement(series([0.36]), duration_sec=2.0)["detected"] is False


def test_movement_bands():
    still = analyse_movement(series([0.36] * 20), duration_sec=2.0)
    assert interpret_movement(still)["gestures"] == "too_still"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_upright_still_body_scores_high():
    pose = analyse_frames(frames(15))
    pose.pop("frames", None)
    scored = score_posture(pose, {"detected": True, "sway": 0.0})
    assert scored["score"] >= 90


def test_bad_posture_scores_lower_than_good():
    good = analyse_frames(frames(10))
    bad = analyse_frames(frames(10, shoulder_dy=0.10, hip_dx=0.18))
    good.pop("frames", None)
    bad.pop("frames", None)

    movement = {"detected": True, "sway": 0.0}
    assert score_posture(bad, movement)["score"] < score_posture(good, movement)["score"]


def test_undetected_body_has_no_score():
    assert score_posture({"detected": False}, {})["score"] is None


def test_only_measured_dimensions_are_scored():
    """A speaker framed chest-up must not be scored on unseen dimensions."""
    pose = {"detected": True, "shoulder_tilt": 0.0, "head_tilt": 0.0,
            "torso_lean": None, "openness": None, "forward_head": None}
    scored = score_posture(pose, {"detected": False})
    assert scored["measured"] == ["alignment"]
    assert "openness" not in scored["breakdown"]


# ---------------------------------------------------------------------------
# Forward head — self-calibrated, never absolute
# ---------------------------------------------------------------------------
def test_head_scale_is_measured_per_frame():
    narrow = analyse_frame(PoseFrame(body(ear_dx=0.045), aspect=1.0))
    wide = analyse_frame(PoseFrame(body(ear_dx=0.060), aspect=1.0))
    assert wide["head_scale"] > narrow["head_scale"]


def test_forward_head_is_unknown_without_a_baseline():
    """
    The old absolute z-depth threshold nagged correctly-seated people —
    webcam perspective puts everyone's ears nearer the camera. With no
    personal baseline the only honest band is "unknown".
    """
    pose = analyse_frames(frames(6))
    pose.pop("frames", None)
    assert interpret(pose)["forward_head"] == "unknown"


def test_unknown_forward_head_is_excluded_from_the_score():
    pose = analyse_frames(frames(6))
    pose.pop("frames", None)
    scored = score_posture(pose, {"detected": True, "sway": 0.0})
    assert "head" not in scored["breakdown"]


def test_session_baseline_flags_drift_not_absolutes():
    from backend.services.session_store import SessionStore

    store = SessionStore()
    session = store.create()
    try:
        assert session.calibrate_head_scale(0.45) == 1.0   # first batch seeds
        assert session.calibrate_head_scale(0.45) == pytest.approx(1.0, abs=0.01)
        craned = session.calibrate_head_scale(0.58)         # head 29% closer
        assert craned > 1.2
        # Sitting back below the old neutral snaps the baseline down.
        assert session.calibrate_head_scale(0.43) == pytest.approx(1.0, abs=0.01)
    finally:
        store.clear()


def test_endpoint_chin_warning_needs_calibrated_drift(client):
    session_id = client.post("/api/session").json()["session_id"]

    def chin_msgs(response):
        return [m for m in response["feedback"] if m["category"] == "neck"]

    # Neutral batch seeds the baseline: no chin nagging on first sight.
    first = client.post("/api/pose",
                        json=pose_payload(session_id=session_id)).json()
    assert not [m for m in chin_msgs(first) if m["type"] == "warning"]

    # Same person, head 33% larger -> drift from their own neutral.
    craned = client.post("/api/pose",
                         json=pose_payload(session_id=session_id, ear_dx=0.060)).json()
    assert craned["bands"]["forward_head"] == "pronounced"
    assert any(m["type"] == "warning" for m in chin_msgs(craned))


def test_endpoint_without_session_never_judges_the_chin(client):
    response = client.post("/api/pose", json=pose_payload(ear_dx=0.060)).json()
    assert response["bands"]["forward_head"] == "unknown"
    assert not [m for m in response["feedback"] if m["category"] == "neck"]


# ---------------------------------------------------------------------------
# Chin false negative: starting the camera already craned
# ---------------------------------------------------------------------------
def test_baseline_ceiling_catches_a_craned_start():
    """
    Pure self-calibration learns whatever it sees first: start craned and
    that pose becomes 'neutral', so it is never flagged. The anatomical
    ceiling (ear span ≤ ~0.52 shoulder widths at true neutral) caps the
    baseline, so an oversized head reads as leaning-in from frame one.
    """
    from backend.services.session_store import SessionStore

    store = SessionStore()
    session = store.create()
    try:
        # First thing the camera ever sees: head at 0.62 shoulder-widths.
        deviation = session.calibrate_head_scale(0.62)
        assert deviation is not None and deviation > 1.15
        assert session.head_scale_base <= 0.52
    finally:
        store.clear()


def test_normal_proportions_are_untouched_by_the_ceiling():
    from backend.services.session_store import SessionStore

    store = SessionStore()
    session = store.create()
    try:
        assert session.calibrate_head_scale(0.45) == 1.0
        assert session.head_scale_base == pytest.approx(0.45)
    finally:
        store.clear()


def test_endpoint_flags_a_session_that_starts_craned(client):
    session_id = client.post("/api/session").json()["session_id"]
    response = client.post(
        "/api/pose", json=pose_payload(session_id=session_id, ear_dx=0.065)
    ).json()
    # ear span 0.13 over shoulder width 0.2 → 0.65: flagged immediately,
    # no sit-back required first.
    assert response["bands"]["forward_head"] in ("noticeable", "pronounced")


# ---------------------------------------------------------------------------
# Hand at the face
# ---------------------------------------------------------------------------
def test_hand_face_distance_is_measured():
    covered = analyse_frame(PoseFrame(body(hand_at_mouth=True), aspect=1.0))
    free = analyse_frame(PoseFrame(body(), aspect=1.0))
    assert covered["hand_face_dist"] < 0.30
    assert free["hand_face_dist"] > 0.30


def test_hand_on_face_ratio_reflects_persistence():
    mixed = [PoseFrame(body(hand_at_mouth=(i < 4)), aspect=1.0) for i in range(10)]
    summary = analyse_frames(mixed)
    assert summary["hand_on_face_ratio"] == pytest.approx(0.4)


def test_hand_on_face_bands():
    covered = analyse_frames(frames(10, hand_at_mouth=True))
    covered.pop("frames", None)
    assert interpret(covered)["hand_on_face"] == "pronounced"

    free = analyse_frames(frames(10))
    free.pop("frames", None)
    assert interpret(free)["hand_on_face"] == "good"


def test_endpoint_warns_about_a_covered_mouth(client):
    response = client.post(
        "/api/pose", json=pose_payload(hand_at_mouth=True)
    ).json()
    hands = [m for m in response["feedback"]
             if m["category"] == "hands" and m["type"] == "warning"]
    assert hands, "a covered mouth must produce a warning"
    assert "face" in hands[0]["text"].lower() or "hand" in hands[0]["text"].lower()


def test_gesturing_hands_do_not_read_as_face_touching(client):
    """Hands at chest height are gesturing space, not face cover."""
    response = client.post("/api/pose", json=pose_payload()).json()
    assert response["bands"]["hand_on_face"] == "good"


# ---------------------------------------------------------------------------
# Presence fusion
# ---------------------------------------------------------------------------
def test_presence_combines_voice_and_body():
    assert presence_score(80, 60) == pytest.approx(72, abs=1)   # .6/.4


def test_presence_falls_back_to_whichever_half_exists():
    """No camera must not read as a 50% body score."""
    assert presence_score(80, None) == 80
    assert presence_score(None, 60) == 60
    assert presence_score(None, None) is None


# ---------------------------------------------------------------------------
# Coaching copy
# ---------------------------------------------------------------------------
def test_feedback_asks_the_speaker_into_frame_when_undetected():
    messages = generate_posture_feedback({"detected": False}, {}, {}, {})
    assert len(messages) == 1
    assert "frame" in messages[0]["text"].lower()


def test_feedback_is_well_formed():
    pose = analyse_frames(frames(10))
    pose.pop("frames", None)
    movement = analyse_movement(series([0.36] * 20), 2.0)
    messages = generate_posture_feedback(
        pose, movement, interpret(pose), interpret_movement(movement)
    )
    assert messages
    for m in messages:
        assert set(m) == {"text", "type", "category"}
        assert m["type"] in {"info", "success", "warning"}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
def pose_payload(count=15, session_id=None, **kwargs):
    return {
        "session_id": session_id,
        "duration": 2.5,
        "aspect": 4 / 3,
        "frames": [{"t": i * 0.1, "landmarks": body(**kwargs)} for i in range(count)],
    }


def test_pose_endpoint_scores_a_batch(client):
    body_json = client.post("/api/pose", json=pose_payload()).json()
    assert body_json["detected"] is True
    assert 0 <= body_json["score"] <= 100
    assert body_json["feedback"]


def test_pose_endpoint_rejects_an_empty_batch(client):
    response = client.post("/api/pose", json=pose_payload(count=0))
    assert response.status_code == 400


def test_pose_endpoint_caps_batch_size(client):
    response = client.post("/api/pose", json=pose_payload(count=400))
    assert response.status_code == 413


def test_pose_endpoint_handles_nobody_in_frame(client):
    response = client.post("/api/pose", json=pose_payload(visible=False)).json()
    assert response["detected"] is False
    assert response["score"] is None


def test_pose_folds_into_the_session(client):
    session_id = client.post("/api/session").json()["session_id"]
    client.post("/api/pose", json=pose_payload(session_id=session_id))
    client.post("/api/pose", json=pose_payload(session_id=session_id))

    summary = client.get(f"/api/session/{session_id}").json()
    assert summary["posture_batches"] == 2
    assert summary["avg_posture"] is not None


def test_pose_reports_an_unknown_session(client):
    body_json = client.post("/api/pose", json=pose_payload(session_id="nope")).json()
    assert "session_not_found" in body_json["warnings"]


def test_presence_appears_once_both_halves_are_seen(client, tone_wav, stub_transcribe):
    session_id = client.post("/api/session").json()["session_id"]

    client.post("/api/analyze",
                files={"audio_file": ("c.wav", tone_wav, "audio/wav")},
                data={"session_id": session_id})
    body_json = client.post("/api/pose", json=pose_payload(session_id=session_id)).json()

    assert body_json["presence_score"] is not None
