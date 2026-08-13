"""Audio decoding, resampling, and acoustic analysis."""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.audio_analysis import (
    analyze_audio,
    compute_energy,
    compute_pitch,
    detect_pauses,
)
from backend.services.audio_io import AudioDecodeError, prepare, resample, to_mono
from backend.utils.helpers import SAMPLE_RATE, rms_to_dbfs

from conftest import silence, sine, to_wav_bytes


# ---------------------------------------------------------------------------
# Decoding & resampling
# ---------------------------------------------------------------------------
def test_prepare_reports_the_source_rate_and_resamples():
    """The bug this guards: sample rate read, then silently discarded."""
    wav = to_wav_bytes(sine(seconds=1.0, sr=48_000), sr=48_000)
    audio, source_sr = prepare(wav)

    assert source_sr == 48_000
    assert audio.dtype == np.float32
    # One second at 48 kHz must become one second at 16 kHz.
    assert len(audio) == pytest.approx(SAMPLE_RATE, rel=0.01)


def test_pitch_is_correct_after_resampling_from_48k():
    """Without resampling this reads ~3x too high."""
    wav = to_wav_bytes(sine(frequency=220.0, seconds=1.0, sr=48_000), sr=48_000)
    audio, _ = prepare(wav)
    assert compute_pitch(audio)["mean_pitch"] == pytest.approx(220.0, abs=6.0)


def test_resample_is_exact_for_integer_ratios():
    audio = sine(seconds=1.0, sr=48_000)
    assert len(resample(audio, 48_000, 16_000)) == 16_000


def test_resample_is_a_no_op_at_the_target_rate():
    audio = sine(seconds=0.5)
    assert resample(audio, SAMPLE_RATE, SAMPLE_RATE) is audio


def test_stereo_is_mixed_to_mono():
    stereo = np.stack([np.ones(100), np.full(100, 3.0)], axis=1)
    assert to_mono(stereo) == pytest.approx(np.full(100, 2.0))


def test_prepare_caps_duration():
    wav = to_wav_bytes(sine(seconds=10.0))
    audio, _ = prepare(wav, max_seconds=2.0)
    assert len(audio) == pytest.approx(2.0 * SAMPLE_RATE, rel=0.01)


def test_prepare_rejects_garbage():
    with pytest.raises(AudioDecodeError):
        prepare(b"this is definitely not audio")


def test_prepare_rejects_empty_payload():
    with pytest.raises(AudioDecodeError):
        prepare(b"")


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def test_energy_of_silence_is_zero():
    assert compute_energy(silence()) == 0.0


def test_energy_matches_the_rms_of_a_sine():
    # RMS of a sine with amplitude a is a / sqrt(2)
    assert compute_energy(sine(amplitude=0.5)) == pytest.approx(0.5 / np.sqrt(2), abs=1e-3)


def test_dbfs_conversion():
    assert rms_to_dbfs(1.0) == pytest.approx(0.0)
    assert rms_to_dbfs(0.1) == pytest.approx(-20.0)
    assert rms_to_dbfs(0.0) == -90.0


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frequency", [110.0, 220.0, 330.0])
def test_pitch_tracks_a_known_tone(frequency):
    result = compute_pitch(sine(frequency=frequency, seconds=1.0))
    assert result["mean_pitch"] == pytest.approx(frequency, rel=0.05)


def test_pitch_of_a_steady_tone_has_low_variation():
    assert compute_pitch(sine(frequency=200.0, seconds=1.0))["pitch_std"] < 5.0


def test_silence_yields_no_pitch():
    result = compute_pitch(silence())
    assert result["mean_pitch"] == 0.0
    assert result["voiced_ratio"] == 0.0


def test_voiced_ratio_reflects_partial_speech():
    """Half tone, half silence -> roughly half the frames are voiced."""
    audio = np.concatenate([sine(seconds=1.0), silence(seconds=1.0)])
    assert compute_pitch(audio)["voiced_ratio"] == pytest.approx(0.5, abs=0.1)


def test_pitch_ignores_frequencies_outside_the_speech_range():
    # 20 Hz sits below PITCH_MIN_HZ and must not be reported.
    assert compute_pitch(sine(frequency=20.0, seconds=1.0))["mean_pitch"] == 0.0


def test_short_input_does_not_crash():
    assert compute_pitch(np.zeros(64, dtype=np.float32))["mean_pitch"] == 0.0


# ---------------------------------------------------------------------------
# Pauses
# ---------------------------------------------------------------------------
def test_all_silence_is_all_pause():
    assert detect_pauses(silence())["pause_ratio"] == 1.0


def test_continuous_tone_has_no_pause():
    assert detect_pauses(sine())["pause_ratio"] == 0.0


def test_longest_pause_is_measured():
    audio = np.concatenate([sine(seconds=0.5), silence(seconds=1.0), sine(seconds=0.5)])
    result = detect_pauses(audio)
    assert result["longest_pause_sec"] == pytest.approx(1.0, abs=0.15)
    assert result["pause_ratio"] == pytest.approx(0.5, abs=0.1)


def test_empty_audio_is_safe():
    assert detect_pauses(np.array([], dtype=np.float32))["pause_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
def test_analyze_audio_returns_every_metric():
    result = analyze_audio(sine())
    assert set(result) == {
        "energy", "energy_db", "mean_pitch", "pitch_std",
        "voiced_ratio", "pause_ratio", "longest_pause_sec",
    }
    assert result["energy_db"] > -30
