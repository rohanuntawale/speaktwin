"""
SpeakTwin - Acoustic Analysis
==============================
Energy, pitch, and pause extraction using numpy + scipy only (no librosa,
which keeps the Windows install painless).

Pitch uses windowed, normalised autocorrelation with three guards the naive
version lacked:
  * voiced-frame gating - silent frames no longer pollute the pitch stats
  * lag-range restriction - only lags inside the human f0 range are searched
  * octave correction - autocorrelation loves period multiples, which read
    as a pitch one octave too low
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np  # type: ignore
from scipy.signal import correlate  # type: ignore

from backend.utils.helpers import (  # type: ignore
    get_logger,
    SAMPLE_RATE,
    FRAME_LENGTH,
    HOP_LENGTH,
    PITCH_MIN_HZ,
    PITCH_MAX_HZ,
    VOICED_FRAME_DBFS,
    VOICING_CORR_THRESHOLD,
    OCTAVE_CORRECTION_RATIO,
    SILENCE_DBFS,
    dbfs_to_rms,
    rms_to_dbfs,
)

logger = get_logger(__name__)

_VOICED_FRAME_RMS = dbfs_to_rms(VOICED_FRAME_DBFS)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def compute_energy(audio: np.ndarray) -> float:
    """Root Mean Square amplitude of the whole chunk."""
    if audio.size == 0:
        return 0.0
    return round(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 6)


def _frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Per-frame RMS via a strided view - no Python-level loop."""
    if audio.size < frame_length:
        return np.array([], dtype=np.float64)

    # as_strided reads raw strides, so the buffer has to be contiguous.
    audio = np.ascontiguousarray(audio)
    num_frames = 1 + (audio.size - frame_length) // hop_length
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(num_frames, frame_length), strides=strides, writeable=False
    )
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------
def _parabolic_peak(values: np.ndarray, index: int) -> float:
    """Sub-sample peak position via parabolic interpolation."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    if denominator == 0:
        return float(index)
    delta = 0.5 * (left - right) / denominator
    if not -1.0 < delta < 1.0:
        return float(index)
    return float(index) + float(delta)


def _is_local_maximum(values: np.ndarray, index: int) -> bool:
    """True when `index` is a real interior peak rather than an edge value."""
    if index <= 0 or index >= len(values) - 1:
        return False
    return values[index] >= values[index - 1] and values[index] >= values[index + 1]


def _octave_correct(norm_corr: np.ndarray, lag: int, min_lag: int) -> int:
    """
    Prefer a sub-multiple lag when it is nearly as strong as the peak.

    Autocorrelation peaks at every multiple of the true period, so the raw
    argmax often lands an octave (or two) below the real pitch.
    """
    peak_strength = norm_corr[lag]
    for divisor in (4, 3, 2):
        candidate = int(round(lag / divisor))
        if candidate < min_lag or candidate >= len(norm_corr):
            continue
        # Take the best value in a small neighbourhood - the exact
        # sub-multiple can sit a sample either side of the local peak.
        lo = max(min_lag, candidate - 2)
        hi = min(len(norm_corr) - 1, candidate + 2)
        local = int(np.argmax(norm_corr[lo:hi + 1])) + lo
        if norm_corr[local] >= OCTAVE_CORRECTION_RATIO * peak_strength:
            return local
    return lag


def compute_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Estimate fundamental frequency across the chunk.

    Returns mean/std of the voiced frames plus the fraction of frames that
    were voiced at all (useful for telling "monotone" from "barely spoke").
    """
    empty = {"mean_pitch": 0.0, "pitch_std": 0.0, "voiced_ratio": 0.0}
    if audio.size < FRAME_LENGTH:
        return empty

    min_lag = max(2, int(sr // PITCH_MAX_HZ))
    max_lag = min(FRAME_LENGTH - 1, int(sr // PITCH_MIN_HZ))
    if max_lag <= min_lag:
        return empty

    window = np.hanning(FRAME_LENGTH)
    f0_values: List[float] = []
    total_frames = 0

    for start in range(0, audio.size - FRAME_LENGTH + 1, HOP_LENGTH):
        total_frames += 1
        frame = audio[start:start + FRAME_LENGTH].astype(np.float64)

        # Voiced-frame gate: silence has no pitch to measure.
        if np.sqrt(np.mean(np.square(frame))) < _VOICED_FRAME_RMS:
            continue

        frame = (frame - frame.mean()) * window
        corr = correlate(frame, frame, mode="full", method="fft")[FRAME_LENGTH - 1:]
        if corr[0] <= 0:
            continue

        norm_corr = corr / corr[0]
        search = norm_corr[min_lag:max_lag + 1]
        if search.size == 0:
            continue

        lag = int(np.argmax(search)) + min_lag
        if norm_corr[lag] < VOICING_CORR_THRESHOLD:
            continue  # unvoiced / noise

        # The peak must be a genuine local maximum. A signal whose true
        # period lies outside the search range (desk rumble, DC drift)
        # still has its highest in-range value pinned to a boundary, and
        # taking that at face value reports the edge of the range as the
        # pitch - a 20 Hz thump reading as a tense 400 Hz voice.
        if not _is_local_maximum(norm_corr, lag):
            continue

        lag = _octave_correct(norm_corr, lag, min_lag)
        refined_lag = _parabolic_peak(norm_corr, lag)
        if refined_lag <= 0:
            continue

        f0 = sr / refined_lag
        if PITCH_MIN_HZ <= f0 <= PITCH_MAX_HZ:
            f0_values.append(f0)

    if not f0_values:
        return empty

    values = np.array(f0_values, dtype=np.float64)

    # Robust mean: drop residual octave outliers before averaging so a
    # handful of bad frames cannot drag the reported pitch around.
    median = float(np.median(values))
    keep = values[(values >= median / 1.6) & (values <= median * 1.6)]
    if keep.size == 0:
        keep = values

    return {
        "mean_pitch": round(float(np.mean(keep)), 2),
        "pitch_std": round(float(np.std(keep)), 2),
        "voiced_ratio": round(len(f0_values) / total_frames, 3) if total_frames else 0.0,
    }


# ---------------------------------------------------------------------------
# Pauses
# ---------------------------------------------------------------------------
def detect_pauses(audio: np.ndarray, sr: int = SAMPLE_RATE,
                  frame_length: int = FRAME_LENGTH,
                  hop_length: int = HOP_LENGTH,
                  silence_dbfs: float = SILENCE_DBFS) -> Dict[str, float]:
    """
    Measure how much of the chunk is silence, and how long the longest
    single stretch of it runs.
    """
    empty = {
        "pause_ratio": 0.0,
        "total_frames": 0,
        "silent_frames": 0,
        "longest_pause_sec": 0.0,
    }
    if audio.size == 0:
        return empty

    rms_values = _frame_rms(audio, frame_length, hop_length)
    total = int(rms_values.size)
    if total == 0:
        return empty

    threshold = dbfs_to_rms(silence_dbfs)
    silent_mask = rms_values < threshold
    silent = int(np.count_nonzero(silent_mask))

    # Longest consecutive run of silent frames.
    longest_run = 0
    current_run = 0
    for is_silent in silent_mask:
        if is_silent:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    seconds_per_frame = hop_length / float(sr) if sr > 0 else 0.0

    return {
        "pause_ratio": round(silent / total, 3),
        "total_frames": total,
        "silent_frames": silent,
        "longest_pause_sec": round(longest_run * seconds_per_frame, 2),
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
def analyze_audio(audio: np.ndarray, sr: int = SAMPLE_RATE,
                  silence_dbfs: float = SILENCE_DBFS) -> Dict[str, float]:
    """Run every acoustic metric over one chunk."""
    energy = compute_energy(audio)
    pitch_info = compute_pitch(audio, sr)
    pause_info = detect_pauses(audio, sr, silence_dbfs=silence_dbfs)

    return {
        "energy": energy,
        "energy_db": round(rms_to_dbfs(energy), 2),
        "mean_pitch": pitch_info["mean_pitch"],
        "pitch_std": pitch_info["pitch_std"],
        "voiced_ratio": pitch_info["voiced_ratio"],
        "pause_ratio": pause_info["pause_ratio"],
        "longest_pause_sec": pause_info["longest_pause_sec"],
    }
