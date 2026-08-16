"""
SpeakTwin - Utility Helpers
============================
Shared constants, threshold configurations, and helper functions
used across the application.

Loudness thresholds are expressed in dBFS rather than raw RMS. A linear
RMS scale makes a "too quiet" rule swing wildly with microphone gain; the
log scale degrades far more gracefully. The RMS equivalents are derived
from the dBFS values so the two can never drift apart.
"""

import logging
import math

from backend.utils.config import get_settings  # type: ignore

# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------
_LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Create a consistent logger for any module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)

    level = getattr(logging, get_settings().log_level, logging.INFO)
    logger.setLevel(level if isinstance(level, int) else logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Audio Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # 16 kHz - optimal for speech recognition
CHUNK_DURATION = 2.5          # seconds per analysis chunk
CHANNELS = 1                  # mono audio

# Frame geometry for acoustic analysis (64 ms window, 32 ms hop @ 16 kHz)
FRAME_LENGTH = 1024
HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# Loudness helpers
# ---------------------------------------------------------------------------
DBFS_FLOOR = -90.0  # reported for digital silence


def rms_to_dbfs(rms: float) -> float:
    """Convert an RMS amplitude (0-1) to dBFS."""
    if rms <= 0:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * math.log10(rms))


def dbfs_to_rms(dbfs: float) -> float:
    """Convert dBFS back to an RMS amplitude."""
    return float(10.0 ** (dbfs / 20.0))


# ---------------------------------------------------------------------------
# Feedback Thresholds
# ---------------------------------------------------------------------------
# Loudness (dBFS)
SILENCE_DBFS = -45.0          # below this -> no speech detected
ENERGY_LOW_DBFS = -36.0       # below this -> speaking too softly
ENERGY_HIGH_DBFS = -12.0      # above this -> speaking too loudly
HARD_SILENCE_FLOOR_DBFS = -60.0   # adaptive gate is never allowed below this
VOICED_FRAME_DBFS = -50.0     # frames quieter than this are skipped for pitch

# Linear RMS equivalents (kept for display and backwards compatibility)
ENERGY_SILENCE_THRESHOLD = dbfs_to_rms(SILENCE_DBFS)      # ~0.0056
ENERGY_LOW_THRESHOLD = dbfs_to_rms(ENERGY_LOW_DBFS)       # ~0.0158
ENERGY_HIGH_THRESHOLD = dbfs_to_rms(ENERGY_HIGH_DBFS)     # ~0.2512

# Pitch (Hz) - typical human speech range
PITCH_MIN_HZ = 60             # search bound for f0 estimation
PITCH_MAX_HZ = 400            # search bound for f0 estimation
PITCH_LOW_THRESHOLD = 100     # monotone / low energy speaking
PITCH_HIGH_THRESHOLD = 320    # tense / stressed
PITCH_VARIATION_LOW = 15      # Hz std-dev -> monotone
PITCH_VARIATION_GOOD = 35     # Hz std-dev -> expressive
VOICING_CORR_THRESHOLD = 0.35  # normalised autocorrelation peak -> voiced
OCTAVE_CORRECTION_RATIO = 0.85  # sub-multiple must be this strong to win

# Words Per Minute
WPM_TOO_SLOW = 100
WPM_TOO_FAST = 175
WPM_OPTIMAL_LOW = 120
WPM_OPTIMAL_HIGH = 160

# ---------------------------------------------------------------------------
# Filler Words
# ---------------------------------------------------------------------------
# Split by how much context is needed to judge them. Hesitation sounds are
# always fillers; discourse markers like "so" or "like" have legitimate
# lexical uses and are only counted when they appear in a filler position.
ALWAYS_FILLERS = [
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "uh-huh",
    "er", "erm", "hmm", "mmm", "ah",
    "you know", "i mean", "kind of", "sort of",
    "basically", "actually", "literally",
]

# phrase -> rule name, resolved in filler_detection._is_filler_here
CONTEXTUAL_FILLERS = {
    "so": "clause_initial",
    "well": "clause_initial",
    "right": "clause_final",
    "like": "like_rule",
}

FILLER_WORDS = ALWAYS_FILLERS + list(CONTEXTUAL_FILLERS)
FILLER_PHRASES = frozenset(FILLER_WORDS)
MAX_FILLER_NGRAM = max(len(f.split()) for f in FILLER_WORDS)

# "like" reads as comparative/verbal - not a filler - after these words
LIKE_LEXICAL_PREV = frozenset({
    "i", "we", "you", "they", "he", "she", "it", "who",
    "would", "'d", "do", "does", "did", "don't", "doesn't", "didn't",
    "really", "just", "much", "more", "something", "nothing", "anything",
    "feel", "feels", "felt", "look", "looks", "looked", "looking",
    "sound", "sounds", "sounded", "seem", "seems", "seemed",
    "taste", "tastes", "smell", "smells", "act", "acts", "acted",
})
# ...or before these
LIKE_LEXICAL_NEXT = frozenset({"to", "this", "that", "these", "those"})

FILLER_RATE_HIGH = 0.08   # fillers per word -> too many fillers

# ---------------------------------------------------------------------------
# Target Keywords (Positive Reinforcement)
# ---------------------------------------------------------------------------
TARGET_KEYWORDS = [
    "ai", "artificial intelligence", "synergy", "solution",
    "innovative", "growth", "strategy", "project",
    "impact", "vision", "efficient", "powerful",
]
MAX_KEYWORD_NGRAM = max(len(k.split()) for k in TARGET_KEYWORDS)

# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------
PAUSE_RATIO_HIGH = 0.6        # above this -> speech feels halting
PAUSE_RATIO_NATURAL = 0.35    # above this -> healthy phrasing

# ---------------------------------------------------------------------------
# Confidence Score Weights
# ---------------------------------------------------------------------------
CONFIDENCE_WEIGHTS = {
    "wpm": 0.25,
    "pitch_variation": 0.25,
    "energy": 0.20,
    "filler_penalty": 0.30,
}

# ---------------------------------------------------------------------------
# Posture & Gesture
# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (33-point model). Named because
# `landmarks[12]` tells a reader nothing.
POSE_NOSE = 0
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LANDMARK_COUNT = 33

# A landmark below this confidence is treated as not seen at all, rather
# than trusted and quietly poisoning an average.
POSE_MIN_VISIBILITY = 0.5
# Fraction of a batch's frames that must be usable before the batch is
# scored. Below this the speaker has probably left the frame.
POSE_MIN_USABLE_FRAMES = 0.4

# Angles in degrees. These are coaching thresholds, not clinical ones:
# the question is "would an audience notice?", not "is this pathological?".
SHOULDER_TILT_NOTICEABLE = 5.0
SHOULDER_TILT_PRONOUNCED = 10.0
HEAD_TILT_NOTICEABLE = 8.0
HEAD_TILT_PRONOUNCED = 15.0
TORSO_LEAN_NOTICEABLE = 7.0
TORSO_LEAN_PRONOUNCED = 14.0

# Head carried forward of the shoulders — the "screen reading" posture.
#
# Detected from apparent head size: craning toward the screen brings only
# the head nearer the camera, so ear-to-ear distance grows RELATIVE to
# shoulder width. Pure xy geometry, where MediaPipe is accurate — unlike
# its z-depth, which is monocular guesswork with a systematic bias (a
# webcam below eye level puts everyone's ears "nearer" than their
# shoulders, so absolute depth thresholds nag correctly-seated people).
#
# Judged against the speaker's own session baseline, never an absolute:
# bodies, chairs, and camera geometry all differ.
HEAD_SCALE_WATCH = 1.10    # head 10% larger than your neutral → noticeable
HEAD_SCALE_FIX = 1.20      # 20% larger → pronounced
HEAD_BASELINE_CREEP = 1.003  # per-batch relaxation so a stale best decays
# Anatomical ceiling on the baseline. Ear-to-ear span runs ~0.33-0.47 of
# shoulder width at neutral for real human proportions, so a baseline
# above this can only mean the speaker was ALREADY leaning in when the
# camera first saw them. Without the ceiling, pure self-calibration
# enshrines that craned pose as "neutral" and can never flag it — the
# false-negative twin of the absolute-threshold false positive.
HEAD_SCALE_CEILING = 0.52

# Hand over the mouth / face — muffles speech and reads as nervous.
# Distance from the nearest hand landmark to the mouth, in shoulder-widths
# (fingertips on the mouth measure ~0-0.15; hands gesturing at chest ~0.8+).
HAND_FACE_RADIUS = 0.30
# Fraction of a batch's frames with a hand at the face before it is called.
HAND_ON_FACE_NOTICEABLE = 0.30
HAND_ON_FACE_PRONOUNCED = 0.60

POSE_MOUTH_LEFT = 9
POSE_MOUTH_RIGHT = 10
# Wrists plus pinky/index/thumb points — everything MediaPipe gives us
# below the wrist.
POSE_HAND_POINTS = (15, 16, 17, 18, 19, 20, 21, 22)

# Legacy z-depth ratio, still *reported* for data purposes but no longer
# used to judge posture (see above for why).
FORWARD_HEAD_NOTICEABLE = 0.18
FORWARD_HEAD_PRONOUNCED = 0.32

# Shoulder width ÷ torso height. Low means hunched or closed off.
OPENNESS_CLOSED = 0.62
OPENNESS_OPEN = 0.85

# Movement of the torso centre across a batch, in shoulder-widths.
SWAY_STEADY = 0.04
SWAY_RESTLESS = 0.12

# Gestures per minute. Speakers are coached toward visible, purposeful
# hands: too few reads as stiff, too many as distracting.
GESTURE_RATE_LOW = 6.0
GESTURE_RATE_HIGH = 40.0
# Wrist travel per frame (shoulder-widths) that counts as gesturing.
GESTURE_MOTION_THRESHOLD = 0.035
# Rapid small movements that read as fidgeting rather than gesturing.
FIDGET_MOTION_THRESHOLD = 0.012

POSTURE_WEIGHTS = {
    "alignment": 0.35,   # shoulders level, head upright, torso vertical
    "head": 0.25,        # not carried forward
    "openness": 0.20,    # not hunched or closed
    "steadiness": 0.20,  # not swaying
}

# How speech and posture combine into one presence score.
PRESENCE_WEIGHTS = {"voice": 0.6, "body": 0.4}
