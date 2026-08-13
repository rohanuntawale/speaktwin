"""
SpeakTwin - Audio Decoding & Normalisation
===========================================
Turns an uploaded audio payload into exactly what the rest of the pipeline
expects: mono, float32, and sampled at `SAMPLE_RATE`.

The previous implementation read the sample rate and discarded it, so every
downstream metric silently assumed 16 kHz. A client sending 44.1 or 48 kHz
audio - which is what a browser produces when it ignores the AudioContext
sampleRate hint - had its WPM and pitch off by roughly 3x.
"""

from __future__ import annotations

import io
import math
from typing import Tuple

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
from scipy.signal import resample_poly  # type: ignore

from backend.utils.helpers import get_logger, SAMPLE_RATE  # type: ignore

logger = get_logger(__name__)


class AudioDecodeError(ValueError):
    """Raised when the uploaded payload is not decodable audio."""


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average multi-channel audio down to a single channel."""
    if audio.ndim == 1:
        return audio
    if audio.shape[1] == 1:
        return audio[:, 0]
    return np.mean(audio, axis=1)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Polyphase-resample `audio` from `src_sr` to `dst_sr`.

    Uses the reduced integer ratio, so common conversions (48k -> 16k) are
    exact rather than approximated.
    """
    if src_sr == dst_sr or audio.size == 0:
        return audio

    divisor = math.gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr) // divisor
    down = int(src_sr) // divisor
    resampled = resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float32)


def decode(data: bytes, max_seconds: float | None = None) -> Tuple[np.ndarray, int]:
    """
    Decode raw audio bytes into (samples, sample_rate).

    Reading through `SoundFile` lets us inspect the header and cap how many
    frames we decode, so an oversized upload cannot be expanded in full
    before we notice.
    """
    if not data:
        raise AudioDecodeError("Empty audio payload")

    try:
        with sf.SoundFile(io.BytesIO(data)) as handle:
            src_sr = int(handle.samplerate)
            if src_sr <= 0:
                raise AudioDecodeError("Audio header reports an invalid sample rate")

            if max_seconds is not None:
                frames = int(max_seconds * src_sr)
                samples = handle.read(frames=frames, dtype="float32", always_2d=True)
                if handle.tell() < handle.frames:
                    logger.info(
                        "Audio truncated to %.1fs (%d of %d frames)",
                        max_seconds, handle.tell(), handle.frames,
                    )
            else:
                samples = handle.read(dtype="float32", always_2d=True)
    except AudioDecodeError:
        raise
    except Exception as exc:  # soundfile raises a variety of RuntimeErrors
        raise AudioDecodeError(f"Unsupported or corrupt audio: {exc}") from exc

    mono = to_mono(np.asarray(samples, dtype=np.float32))
    if mono.size == 0:
        raise AudioDecodeError("Audio contains no samples")

    return mono, src_sr


def prepare(data: bytes, max_seconds: float | None = None,
            target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Full ingest path: decode -> mono -> float32 -> resample -> sanitise.

    Returns (samples_at_target_sr, original_sample_rate).
    """
    mono, src_sr = decode(data, max_seconds=max_seconds)
    audio = resample(mono, src_sr, target_sr)

    # NaN/Inf from a malformed float stream would poison every metric.
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    return audio, src_sr


def duration_seconds(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Length of `audio` in seconds."""
    if sr <= 0:
        return 0.0
    return float(len(audio)) / float(sr)
