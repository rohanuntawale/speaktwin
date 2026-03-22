import numpy as np
from scipy.signal import correlate

from backend.utils.helpers import (
    get_logger,
    SAMPLE_RATE,
    ENERGY_SILENCE_THRESHOLD,
)

logger = get_logger(__name__)


def compute_energy(audio: np.ndarray) -> float:
    """Compute the Root Mean Square (RMS) energy using numpy."""
    if len(audio) == 0: return 0.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return round(rms, 6)


def compute_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Estimate fundamental frequency using autocorrelation (robust alternative to librosa).
    """
    if len(audio) < 512:
        return {"mean_pitch": 0.0, "pitch_std": 0.0}

    # Frame-based pitch estimation
    frame_size = 1024
    hop_size = 512
    pitches = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        # Standard autocorrelation-based pitch detection
        corr = correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find peaks in the autocorrelation
        d = np.diff(corr)
        start = np.where(d < 0)[0]
        if len(start) == 0: continue
        
        peak_idx = np.argmax(corr[start[0]:]) + start[0]
        if peak_idx > 0:
            freq = sr / peak_idx
            if 60 < freq < 500:  # Valid human speech range
                pitches.append(freq)

    if pitches:
        return {
            "mean_pitch": round(float(np.mean(pitches)), 2),
            "pitch_std": round(float(np.std(pitches)), 2),
        }
    return {"mean_pitch": 0.0, "pitch_std": 0.0}


def detect_pauses(audio: np.ndarray, sr: int = SAMPLE_RATE,
                  frame_length: int = 1024, hop_length: int = 512) -> dict:
    """Detect silent frames using fixed-frame RMS calculation."""
    if len(audio) == 0:
        return {"pause_ratio": 0.0, "total_frames": 0, "silent_frames": 0}

    # Simple frame-by-frame energy
    num_frames = (len(audio) - frame_length) // hop_length + 1
    rms_values = []
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms_values.append(np.sqrt(np.mean(frame**2)))
    
    rms_values = np.array(rms_values)
    total = len(rms_values)
    silent = int(np.sum(rms_values < ENERGY_SILENCE_THRESHOLD))
    ratio = round(silent / total, 3) if total > 0 else 0.0

    return {
        "pause_ratio": ratio,
        "total_frames": total,
        "silent_frames": silent,
    }


def analyze_audio(audio: np.ndarray) -> dict:
    """Combines metrics without using librosa."""
    energy = compute_energy(audio)
    pitch_info = compute_pitch(audio)
    pause_info = detect_pauses(audio)

    return {
        "energy": energy,
        "mean_pitch": pitch_info["mean_pitch"],
        "pitch_std": pitch_info["pitch_std"],
        "pause_ratio": pause_info["pause_ratio"],
    }
