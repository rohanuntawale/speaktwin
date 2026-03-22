"""
SpeakTwin - Audio Capture Service
===================================
Manages microphone input using sounddevice with a streaming callback.
Audio is captured in a non-blocking fashion and buffered for analysis.
Uses a singleton pattern so only one capture session exists at a time.
"""

import threading
import numpy as np
import sounddevice as sd
from collections import deque

from backend.utils.helpers import (
    get_logger,
    SAMPLE_RATE,
    CHUNK_DURATION,
    CHANNELS,
)

logger = get_logger(__name__)


class AudioCapture:
    """
    Singleton audio capture manager.
    
    Records microphone input via a non-blocking InputStream and stores
    audio chunks in a thread-safe deque for downstream consumers.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Enforce singleton – only one capture instance across the app."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.sample_rate = SAMPLE_RATE
        self.chunk_duration = CHUNK_DURATION
        self.channels = CHANNELS
        self.block_size = int(self.sample_rate * self.chunk_duration)

        # Store the last N chunks for analysis (ring buffer)
        self._buffer: deque = deque(maxlen=5)
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._accumulated = np.array([], dtype=np.float32)
        self._acc_lock = threading.Lock()

        logger.info(
            "AudioCapture initialised  [SR=%d, chunk=%.1fs, block=%d]",
            self.sample_rate, self.chunk_duration, self.block_size,
        )

    # ------------------------------------------------------------------
    # Callback (runs on audio thread – must be fast, no allocations)
    # ------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Called by sounddevice for every block of incoming audio."""
        if status:
            logger.warning("Audio status: %s", status)

        chunk = indata[:, 0].copy()  # mono, float32

        with self._acc_lock:
            self._accumulated = np.concatenate([self._accumulated, chunk])

            # When we've accumulated enough samples for a full chunk
            while len(self._accumulated) >= self.block_size:
                full_chunk = self._accumulated[:self.block_size]
                self._buffer.append(full_chunk)
                self._accumulated = self._accumulated[self.block_size:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin recording from the default microphone."""
        if self._recording:
            logger.info("Already recording – skipping start()")
            return

        self._buffer.clear()
        self._accumulated = np.array([], dtype=np.float32)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,  # small blocks → low latency
            callback=self._audio_callback,
        )
        self._stream.start()
        self._recording = True
        logger.info("🎙️  Microphone recording started")

    def stop(self) -> None:
        """Stop recording and release the microphone."""
        if not self._recording:
            return
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._recording = False
        logger.info("⏹️  Microphone recording stopped")

    def get_latest_chunk(self) -> np.ndarray | None:
        """
        Return the most recent full audio chunk.
        Returns None if no chunk is available yet.
        """
        if not self._buffer:
            return None
        return self._buffer[-1]

    def get_all_chunks(self) -> list[np.ndarray]:
        """Return all buffered chunks (up to maxlen)."""
        return list(self._buffer)

    @property
    def is_recording(self) -> bool:
        return self._recording
