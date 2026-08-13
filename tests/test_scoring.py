"""Confidence scoring and the rule-based feedback engine."""

from __future__ import annotations

import pytest

from backend.services.confidence_score import calculate_confidence
from backend.services.feedback_engine import generate_feedback
from backend.utils.helpers import (
    ENERGY_HIGH_DBFS,
    ENERGY_LOW_DBFS,
    SILENCE_DBFS,
    rms_to_dbfs,
)

IDEAL_DB = -24.0  # comfortably inside the good loudness band


def score(**overrides) -> dict:
    kwargs = {"wpm": 140.0, "pitch_std": 40.0, "energy_db": IDEAL_DB, "filler_rate": 0.0}
    kwargs.update(overrides)
    return calculate_confidence(**kwargs)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def test_ideal_delivery_scores_full_marks():
    assert score()["score"] == 100


def test_score_is_always_in_range():
    assert score(wpm=0, pitch_std=0, energy_db=-90, filler_rate=1.0)["score"] == 0
    assert 0 <= score(wpm=400, pitch_std=200, energy_db=0, filler_rate=0.5)["score"] <= 100


def test_breakdown_components_are_percentages():
    breakdown = score()["breakdown"]
    assert set(breakdown) == {"wpm", "pitch_variation", "energy", "filler_usage"}
    assert all(0 <= v <= 100 for v in breakdown.values())


@pytest.mark.parametrize("wpm,expected", [(140, 100), (90, 50), (60, 0), (0, 0)])
def test_wpm_subscore_falls_off_outside_the_band(wpm, expected):
    assert score(wpm=wpm)["breakdown"]["wpm"] == pytest.approx(expected, abs=2)


def test_monotone_delivery_is_penalised():
    assert score(pitch_std=2.0)["breakdown"]["pitch_variation"] < 20


def test_fillers_reduce_the_score_monotonically():
    scores = [score(filler_rate=r)["score"] for r in (0.0, 0.05, 0.10, 0.20)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("energy_db,expected", [
    (SILENCE_DBFS - 1, 0),        # below silence
    (ENERGY_LOW_DBFS, 100),       # bottom of the good band
    (IDEAL_DB, 100),              # inside the good band
    (ENERGY_HIGH_DBFS + 10, 40),  # well past loud, floors out
])
def test_energy_subscore_ramps_instead_of_stepping(energy_db, expected):
    assert score(energy_db=energy_db)["breakdown"]["energy"] == pytest.approx(expected, abs=2)


def test_energy_ramp_is_gradual_between_silence_and_the_good_band():
    """A speaker drifting quiet should slide, not fall off a cliff."""
    midpoint = (SILENCE_DBFS + ENERGY_LOW_DBFS) / 2
    assert 30 < score(energy_db=midpoint)["breakdown"]["energy"] < 70


def test_linear_rms_fallback_matches_the_dbfs_path():
    rms = 0.05
    assert (
        calculate_confidence(wpm=140, pitch_std=40, filler_rate=0.0, energy=rms)["score"]
        == calculate_confidence(
            wpm=140, pitch_std=40, filler_rate=0.0, energy_db=rms_to_dbfs(rms)
        )["score"]
    )


# ---------------------------------------------------------------------------
# Feedback engine
# ---------------------------------------------------------------------------
def feedback(**overrides) -> dict:
    kwargs = {
        "energy_db": IDEAL_DB, "mean_pitch": 180.0, "pitch_std": 40.0,
        "wpm": 140.0, "filler_rate": 0.0, "pause_ratio": 0.1,
    }
    kwargs.update(overrides)
    return generate_feedback(**kwargs)


def test_good_delivery_reads_as_excellent():
    result = feedback()
    assert result["status"] == "excellent"
    assert not [m for m in result["messages"] if m["type"] == "warning"]


def test_silence_is_reported_as_silent():
    assert feedback(energy_db=-80, mean_pitch=0.0, pitch_std=0.0,
                    wpm=0.0, pause_ratio=1.0)["status"] == "silent"


def test_pitch_messages_are_skipped_when_nothing_was_voiced():
    categories = {m["category"] for m in feedback(mean_pitch=0.0)["messages"]}
    assert "pitch" not in categories


def test_poor_delivery_accumulates_warnings():
    result = feedback(energy_db=-40, mean_pitch=350.0, pitch_std=5.0,
                      wpm=220.0, filler_rate=0.3, pause_ratio=0.9)
    warnings = [m for m in result["messages"] if m["type"] == "warning"]
    assert len(warnings) >= 4
    assert result["status"] == "poor"


def test_every_message_is_well_formed():
    for message in feedback()["messages"]:
        assert set(message) == {"text", "type", "category"}
        assert message["type"] in {"info", "success", "warning"}
        assert message["text"]
