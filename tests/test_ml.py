"""
The ML layer's contract.

The point of these tests is that the neural models are *optional*: with
none installed or enabled, the backend must behave exactly as it did
before. Tests that need a real model are skipped rather than failed, so
the suite stays green on a base install and in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.ml import disfluency
from backend.services.ml.registry import ModelRegistry

from conftest import sine

SR = 16_000


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(not torch_available(), reason="torch not installed")


# ---------------------------------------------------------------------------
# Disabled-by-default contract
# ---------------------------------------------------------------------------
def test_all_ml_is_off_by_default():
    """A base install must not silently start loading neural models."""
    from backend.utils.config import get_settings

    settings = get_settings()
    assert settings.any_ml_enabled is False
    assert settings.ml_pitch_enabled is False
    assert settings.ml_vad_enabled is False
    assert settings.ml_disfluency_enabled is False


def test_disabled_services_return_none():
    from backend.services.ml import alignment, emotion, prosody, speaker
    from backend.services.ml import pitch as ml_pitch
    from backend.services.ml import vad

    audio = sine()
    assert ml_pitch.estimate_pitch(audio) is None
    assert vad.detect_speech(audio) is None
    assert emotion.analyze(audio) is None
    assert prosody.extract(audio) is None
    assert speaker.embed(audio) is None
    assert alignment.align(audio, "some words") is None
    assert disfluency.detect(audio) is None


def test_analysis_response_omits_ml_fields_when_off(client, tone_wav, stub_transcribe):
    body = client.post(
        "/api/analyze",
        files={"audio_file": ("chunk.wav", tone_wav, "audio/wav")},
    ).json()

    for field in ("emotion", "prosody", "disfluency", "alignment",
                  "pitch_confidence", "speech_ratio"):
        assert body[field] is None, f"{field} should be null when ML is off"


def test_health_reports_ml_state(client):
    ml = client.get("/api/health").json()["ml"]
    assert ml["any_enabled"] is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_reports_a_missing_dependency_instead_of_raising():
    registry = ModelRegistry()

    def loader():
        raise ImportError("no module named 'definitely_not_installed'")

    registry.register("fake", "Fake model", "requirements-ml.txt", loader)

    assert registry.get("fake") is None
    status = registry.status()["models"][0]
    assert status["loaded"] is False
    assert status["load_attempted"] is True
    assert "missing dependency" in status["error"]
    assert "requirements-ml.txt" in status["error"]


def test_registry_does_not_retry_a_failed_load():
    """A model that failed once must not be retried on every request."""
    registry = ModelRegistry()
    calls = []

    def loader():
        calls.append(1)
        raise RuntimeError("boom")

    registry.register("flaky", "Flaky", "requirements-ml.txt", loader)
    for _ in range(5):
        registry.get("flaky")

    assert len(calls) == 1


def test_registry_loads_once_and_caches():
    registry = ModelRegistry()
    calls = []

    def loader():
        calls.append(1)
        return object()

    registry.register("once", "Once", "requirements-ml.txt", loader)
    first = registry.get("once")
    second = registry.get("once")

    assert first is second
    assert len(calls) == 1


def test_registry_reset_allows_reload():
    registry = ModelRegistry()
    calls = []
    registry.register("r", "R", "x", lambda: calls.append(1) or object())

    registry.get("r")
    registry.reset("r")
    registry.get("r")

    assert len(calls) == 2


def test_unknown_model_key_is_safe():
    assert ModelRegistry().get("nope") is None


# ---------------------------------------------------------------------------
# Disfluency merge logic (pure, no model needed)
# ---------------------------------------------------------------------------
def text_result(total=2, details=None, words=20):
    return {
        "total_fillers": total,
        "filler_rate": round(total / words, 4),
        "details": details if details is not None else {"um": 2},
        "total_words": words,
    }


def test_merge_is_a_no_op_without_an_acoustic_result():
    original = text_result()
    assert disfluency.merge_with_text(original, None) is original


def test_merge_takes_the_maximum_not_the_sum():
    """
    Whisper sometimes keeps an 'um'. When it does, both detectors see the
    same event, so summing would double-count it.
    """
    merged = disfluency.merge_with_text(
        text_result(total=2, details={"um": 2}),
        {"total_fillers": 2, "details": {"um": 2}, "events": [], "engine": "acoustic"},
    )
    assert merged["total_fillers"] == 2
    assert merged["details"]["um"] == 2


def test_merge_surfaces_fillers_whisper_deleted():
    """The whole reason the acoustic detector exists."""
    merged = disfluency.merge_with_text(
        text_result(total=0, details={}, words=20),
        {"total_fillers": 5, "details": {"um": 3, "uh": 2}, "events": [],
         "engine": "acoustic"},
    )
    assert merged["total_fillers"] == 5
    assert merged["details"] == {"um": 3, "uh": 2}
    assert merged["filler_rate"] == 0.25
    assert merged["text_fillers"] == 0
    assert merged["acoustic_fillers"] == 5


def test_merge_combines_per_label():
    merged = disfluency.merge_with_text(
        text_result(total=1, details={"um": 1}),
        {"total_fillers": 3, "details": {"uh": 3}, "events": [], "engine": "acoustic"},
    )
    assert merged["details"] == {"um": 1, "uh": 3}


# ---------------------------------------------------------------------------
# Frame-to-event collapsing (pure)
# ---------------------------------------------------------------------------
def frames(sequence, confidence=0.9):
    """Build a (frames, classes) probability matrix from label indices."""
    matrix = np.zeros((len(sequence), len(disfluency.LABELS)), dtype=np.float32)
    for i, label_id in enumerate(sequence):
        matrix[i, label_id] = confidence
        matrix[i, 0] = 1.0 - confidence if label_id != 0 else confidence
    return matrix


def test_consecutive_frames_collapse_to_one_event():
    """40 frames of 'um' is one filler, not 40."""
    result = disfluency._events_from_frames(
        frames([0, 0, 1, 1, 1, 1, 0, 0]),
        disfluency.LABELS, duration=2.5, threshold=0.5,
    )
    assert result["total_fillers"] == 1
    assert result["details"] == {"um": 1}


def test_separated_events_are_counted_separately():
    result = disfluency._events_from_frames(
        frames([1, 1, 0, 0, 0, 2, 2]),
        disfluency.LABELS, duration=2.5, threshold=0.5,
    )
    assert result["total_fillers"] == 2
    assert result["details"] == {"um": 1, "uh": 1}


def test_low_confidence_frames_are_discarded():
    result = disfluency._events_from_frames(
        frames([1, 1, 1], confidence=0.2),
        disfluency.LABELS, duration=2.5, threshold=0.5,
    )
    assert result["total_fillers"] == 0


def test_all_negative_frames_produce_no_events():
    result = disfluency._events_from_frames(
        frames([0] * 10), disfluency.LABELS, duration=2.5, threshold=0.5
    )
    assert result["total_fillers"] == 0
    assert result["events"] == []


def test_event_timings_are_derived_from_frame_positions():
    result = disfluency._events_from_frames(
        frames([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
        disfluency.LABELS, duration=1.0, threshold=0.5,
    )
    event = result["events"][0]
    assert event["start"] == pytest.approx(0.2, abs=0.01)
    assert event["end"] == pytest.approx(0.4, abs=0.01)


def test_empty_frame_matrix_is_safe():
    result = disfluency._events_from_frames(
        np.zeros((0, len(disfluency.LABELS)), dtype=np.float32),
        disfluency.LABELS, duration=2.5, threshold=0.5,
    )
    assert result["total_fillers"] == 0


# ---------------------------------------------------------------------------
# Emotion label mapping
# ---------------------------------------------------------------------------
def test_abbreviated_emotion_labels_are_mapped():
    """
    Regression: `superb/wav2vec2-base-superb-er` emits IEMOCAP-style
    abbreviations, not full words. Matching only the long spelling scored a
    delivery that was 98% angry as tension 0.003.
    """
    from backend.services.ml.emotion import POSITIVE_EMOTIONS, TENSE_EMOTIONS

    for short, long in (("ang", "angry"), ("sad", "sadness"), ("fea", "fear")):
        assert short in TENSE_EMOTIONS, f"{short} missing from TENSE_EMOTIONS"
        assert long in TENSE_EMOTIONS

    assert "hap" in POSITIVE_EMOTIONS
    assert "happy" in POSITIVE_EMOTIONS


def test_emotion_buckets_do_not_overlap():
    from backend.services.ml.emotion import (
        NEUTRAL_EMOTIONS,
        POSITIVE_EMOTIONS,
        TENSE_EMOTIONS,
    )

    assert not TENSE_EMOTIONS & POSITIVE_EMOTIONS
    assert not TENSE_EMOTIONS & NEUTRAL_EMOTIONS
    assert not POSITIVE_EMOTIONS & NEUTRAL_EMOTIONS


def test_superb_label_set_is_fully_mapped():
    """Every label the default checkpoint emits must land in a bucket."""
    from backend.services.ml.emotion import (
        NEUTRAL_EMOTIONS,
        POSITIVE_EMOTIONS,
        TENSE_EMOTIONS,
    )

    known = TENSE_EMOTIONS | POSITIVE_EMOTIONS | NEUTRAL_EMOTIONS
    for label in ("ang", "hap", "neu", "sad"):
        assert label in known, f"{label} unmapped - tension would understate"


# ---------------------------------------------------------------------------
# Real model checks - skipped unless torch is installed
# ---------------------------------------------------------------------------
@requires_torch
def test_registry_resolves_a_torch_device():
    device = ModelRegistry().device()
    assert device in {"cpu", "cuda", "mps"}


@requires_torch
def test_crepe_tracks_a_known_tone(monkeypatch):
    """Only runs when torchcrepe is importable; otherwise skipped."""
    pytest.importorskip("torchcrepe")

    import dataclasses

    from backend.services.ml import pitch as ml_pitch
    from backend.utils.config import get_settings

    enabled = dataclasses.replace(get_settings(), ml_pitch_enabled=True)
    monkeypatch.setattr("backend.services.ml.pitch.get_settings", lambda: enabled)
    monkeypatch.setattr("backend.services.ml.registry.get_settings", lambda: enabled)

    result = ml_pitch.estimate_pitch(sine(frequency=220.0, seconds=1.0))
    if result is None:
        pytest.skip("CREPE weights unavailable")

    assert result["mean_pitch"] == pytest.approx(220.0, rel=0.05)
    assert result["engine"] == "crepe"
