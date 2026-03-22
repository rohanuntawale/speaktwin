"""
SpeakTwin - Utility Helpers
============================
Shared constants, threshold configurations, and helper functions
used across the application.
"""

import logging

# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Create a consistent logger for any module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Audio Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # 16 kHz - optimal for speech recognition
CHUNK_DURATION = 2.5          # seconds per analysis chunk
CHANNELS = 1                  # mono audio

# ---------------------------------------------------------------------------
# Feedback Thresholds
# ---------------------------------------------------------------------------
# Energy (RMS amplitude)
ENERGY_SILENCE_THRESHOLD = 0.005   # below this → no speech detected
ENERGY_LOW_THRESHOLD = 0.015       # below this → speaking too softly
ENERGY_HIGH_THRESHOLD = 0.25       # above this → speaking too loudly

# Pitch (Hz) - typical human speech range
PITCH_LOW_THRESHOLD = 100          # monotone / low energy speaking
PITCH_HIGH_THRESHOLD = 320         # tense / stressed
PITCH_VARIATION_LOW = 15           # Hz std-dev → monotone
PITCH_VARIATION_GOOD = 35          # Hz std-dev → expressive

# Words Per Minute
WPM_TOO_SLOW = 100
WPM_TOO_FAST = 175
WPM_OPTIMAL_LOW = 120
WPM_OPTIMAL_HIGH = 160

# Filler Words
FILLER_WORDS = [
    "um", "uh", "uh-huh", "uhh", "umm",
    "like", "basically", "you know", "actually",
    "literally", "right", "so", "well",
    "kind of", "sort of", "i mean",
]
FILLER_RATE_HIGH = 0.08   # fillers per word → too many fillers

# Target Keywords (Positive Reinforcement)
TARGET_KEYWORDS = [
    "ai", "artificial intelligence", "synergy", "solution",
    "innovative", "growth", "strategy", "project", 
    "impact", "vision", "efficient", "powerful"
]

# Confidence Score Weights
CONFIDENCE_WEIGHTS = {
    "wpm": 0.25,
    "pitch_variation": 0.25,
    "energy": 0.20,
    "filler_penalty": 0.30,
}
