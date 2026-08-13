"""Session aggregation, smoothing, and the adaptive silence gate."""

from __future__ import annotations

import time

import pytest

from backend.services.session_store import (
    MIN_CHUNKS_FOR_ADAPTIVE_GATE,
    Session,
    SessionStore,
)
from backend.utils.helpers import HARD_SILENCE_FLOOR_DBFS, SILENCE_DBFS


def chunk(transcript="we build innovative solutions", confidence=80,
          wpm=140.0, energy_db=-24.0, fillers=1, keywords=1, words=4) -> dict:
    return {
        "transcript": transcript,
        "energy_db": energy_db,
        "confidence_score": confidence,
        "wpm": wpm,
        "clarity": 75,
        "pitch": 180.0,
        "fillers": {
            "total_fillers": fillers,
            "filler_rate": 0.1,
            "details": {"um": fillers} if fillers else {},
            "total_words": words,
        },
        "keywords": {
            "total_keywords": keywords,
            "found_keywords": {"innovative": keywords} if keywords else {},
            "keywords_list": ["innovative"] if keywords else [],
        },
    }


@pytest.fixture
def store():
    store = SessionStore()
    yield store
    store.clear()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_create_and_fetch(store):
    session = store.create()
    assert store.get(session.session_id) is session
    assert store.active_count() == 1


def test_unknown_session_returns_none(store):
    assert store.get("nope") is None
    assert store.get(None) is None
    assert store.record("nope", chunk(), 2.5) is None
    assert store.end("nope") is None


def test_end_returns_the_report_and_frees_the_slot(store):
    session = store.create()
    store.record(session.session_id, chunk(), 2.5)

    report = store.end(session.session_id)
    assert report["total_fillers"] == 1
    assert "transcript" in report
    assert store.active_count() == 0


def test_expired_sessions_are_dropped_on_access(store):
    session = store.create()
    session.updated_at = time.time() - 10_000
    assert store.get(session.session_id) is None
    assert store.active_count() == 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_totals_accumulate_across_chunks(store):
    session = store.create()
    for _ in range(3):
        summary = store.record(session.session_id, chunk(), 2.5)

    assert summary["chunk_count"] == 3
    assert summary["analysed_chunks"] == 3
    assert summary["total_fillers"] == 3
    assert summary["total_words"] == 12
    assert summary["filler_details"] == {"um": 3}
    assert summary["keyword_details"] == {"innovative": 3}
    assert summary["audio_seconds"] == pytest.approx(7.5)


def test_silent_chunks_count_but_do_not_pollute_averages(store):
    session = store.create()
    store.record(session.session_id, chunk(), 2.5)
    summary = store.record(session.session_id, chunk(transcript="", energy_db=-70), 2.5)

    assert summary["chunk_count"] == 2
    assert summary["analysed_chunks"] == 1
    assert summary["total_words"] == 4


def test_transcript_segments_are_stitched(store):
    session = store.create()
    store.record(session.session_id, chunk(transcript="first part"), 2.5)
    store.record(session.session_id, chunk(transcript="second part"), 2.5)
    assert session.transcript == "first part second part"


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def test_first_chunk_seeds_the_average(store):
    session = store.create()
    summary = store.record(session.session_id, chunk(confidence=80), 2.5)
    assert summary["avg_confidence"] == 80


def test_smoothing_damps_a_single_outlier(store):
    session = store.create()
    for _ in range(5):
        store.record(session.session_id, chunk(confidence=80), 2.5)
    summary = store.record(session.session_id, chunk(confidence=10), 2.5)

    # The raw score dropped to 10; the trend must not follow it all the way.
    assert 40 < summary["avg_confidence"] < 80


def test_peak_confidence_is_the_raw_maximum(store):
    session = store.create()
    store.record(session.session_id, chunk(confidence=55), 2.5)
    store.record(session.session_id, chunk(confidence=95), 2.5)
    summary = store.record(session.session_id, chunk(confidence=60), 2.5)
    assert summary["peak_confidence"] == 95


# ---------------------------------------------------------------------------
# Decoder context
# ---------------------------------------------------------------------------
def test_prompt_context_is_empty_before_any_speech():
    session = Session(session_id="s", created_at=0.0, updated_at=0.0)
    assert session.prompt_context() is None


def test_prompt_context_returns_the_transcript_tail(store):
    session = store.create()
    store.record(session.session_id, chunk(transcript="a" * 400), 2.5)
    context = session.prompt_context()
    assert context and len(context) <= 220


# ---------------------------------------------------------------------------
# Adaptive silence gate
# ---------------------------------------------------------------------------
def test_gate_uses_the_absolute_threshold_until_enough_history(store):
    session = store.create()
    assert session.silence_gate_db() == SILENCE_DBFS


def test_quiet_microphone_lowers_the_gate(store):
    """A low-gain mic must not have all of its speech treated as silence."""
    session = store.create()
    for _ in range(MIN_CHUNKS_FOR_ADAPTIVE_GATE + 2):
        store.record(session.session_id, chunk(transcript="", energy_db=-62.0), 2.5)
    assert session.silence_gate_db() < SILENCE_DBFS


def test_loud_room_can_never_raise_the_gate(store):
    """Otherwise background noise would start getting transcribed."""
    session = store.create()
    for _ in range(MIN_CHUNKS_FOR_ADAPTIVE_GATE + 2):
        store.record(session.session_id, chunk(energy_db=-10.0), 2.5)
    assert session.silence_gate_db() == SILENCE_DBFS


def test_gate_never_drops_below_the_hard_floor(store):
    session = store.create()
    for _ in range(MIN_CHUNKS_FOR_ADAPTIVE_GATE + 2):
        store.record(session.session_id, chunk(transcript="", energy_db=-90.0), 2.5)
    assert session.silence_gate_db() >= HARD_SILENCE_FLOOR_DBFS


# ---------------------------------------------------------------------------
# Speaker continuity
# ---------------------------------------------------------------------------
THRESHOLD = 0.6
VOICE_A = [1.0, 0.0, 0.0, 0.0]
VOICE_A_ISH = [0.95, 0.31, 0.0, 0.0]   # same speaker, slight variation
VOICE_B = [0.0, 1.0, 0.0, 0.0]         # orthogonal - clearly different


def test_first_embedding_seeds_the_reference(store):
    session = store.create()
    assert session.track_speaker(VOICE_A, THRESHOLD) == 1.0
    assert session.reference_embedding == VOICE_A


def test_missing_embedding_is_ignored(store):
    session = store.create()
    assert session.track_speaker(None, THRESHOLD) is None
    assert session.reference_embedding is None


def test_same_speaker_scores_high(store):
    session = store.create()
    session.track_speaker(VOICE_A, THRESHOLD)
    assert session.track_speaker(VOICE_A_ISH, THRESHOLD) > THRESHOLD
    assert session.speaker_changes == 0


def test_different_speaker_is_detected(store):
    session = store.create()
    session.track_speaker(VOICE_A, THRESHOLD)
    assert session.track_speaker(VOICE_B, THRESHOLD) < THRESHOLD
    assert session.speaker_changes == 1


def test_speaker_changes_counts_transitions_not_frames(store):
    """A second speaker talking for six chunks is one change, not six."""
    session = store.create()
    session.track_speaker(VOICE_A, THRESHOLD)
    for _ in range(6):
        session.track_speaker(VOICE_B, THRESHOLD)
    assert session.speaker_changes == 1


def test_switching_back_and_forth_counts_each_transition(store):
    session = store.create()
    session.track_speaker(VOICE_A, THRESHOLD)
    session.track_speaker(VOICE_B, THRESHOLD)   # change 1
    session.track_speaker(VOICE_A, THRESHOLD)   # back to reference
    session.track_speaker(VOICE_B, THRESHOLD)   # change 2
    assert session.speaker_changes == 2


def test_speaker_fields_appear_in_the_summary(store):
    session = store.create()
    session.track_speaker(VOICE_A, THRESHOLD)
    summary = store.record(session.session_id, chunk(), 2.5)
    assert summary["speaker_similarity"] == 1.0
    assert summary["speaker_changes"] == 0


def test_speaker_fields_are_null_when_tracking_never_ran(store):
    session = store.create()
    summary = store.record(session.session_id, chunk(), 2.5)
    assert summary["speaker_similarity"] is None
    assert summary["speaker_changes"] == 0


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------
def test_store_evicts_oldest_when_over_capacity(store, monkeypatch):
    import dataclasses

    from backend.utils.config import get_settings

    capped = dataclasses.replace(get_settings(), max_sessions=3)
    monkeypatch.setattr(
        "backend.services.session_store.get_settings", lambda: capped
    )

    created = [store.create() for _ in range(5)]
    assert store.active_count() == 3
    # The most recent survives; the oldest were evicted.
    assert store.get(created[-1].session_id) is not None
    assert store.get(created[0].session_id) is None
