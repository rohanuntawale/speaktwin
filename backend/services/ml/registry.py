"""
SpeakTwin - ML Model Registry
==============================
One place that owns loading, caching, and reporting on every deep-learning
model in the backend.

Design rules, all of which follow from the fact that SpeakTwin analyses a
2.5s chunk every 2.5s:

  * **Lazy.** Nothing is imported or downloaded until the feature is
    actually enabled and first used. Importing torch alone costs seconds.
  * **Once.** Each model loads a single time behind a lock, exactly like
    the Whisper singleton in `speech_to_text`.
  * **Optional.** Every dependency is soft. A missing package disables one
    feature and reports why - it never breaks the request. This is what
    keeps `pip install -r requirements.txt` (no ML extras) a valid install.
  * **Serialised.** Torch modules are not safe to call concurrently from
    the threadpool, so inference holds a per-model lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from backend.utils.config import get_settings  # type: ignore
from backend.utils.helpers import get_logger  # type: ignore

logger = get_logger(__name__)


@dataclass
class ModelSlot:
    """Bookkeeping for one lazily-loaded model."""

    key: str
    description: str
    loader: Callable[[], Any]
    extra: str                      # pip extra that provides it

    model: Any = None
    load_attempted: bool = False
    error: Optional[str] = None
    load_seconds: float = 0.0
    load_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    infer_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def status(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "loaded": self.loaded,
            "load_attempted": self.load_attempted,
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds else None,
            "error": self.error,
            "install_extra": self.extra,
        }


class ModelRegistry:
    """Process-wide registry of optional ML models."""

    def __init__(self) -> None:
        self._slots: Dict[str, ModelSlot] = {}
        self._lock = threading.RLock()
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    def device(self) -> str:
        """
        Resolve the torch device once.

        `auto` prefers CUDA, then Apple Silicon MPS, then CPU.
        """
        if self._device is not None:
            return self._device

        configured = get_settings().ml_device.lower()
        if configured != "auto":
            self._device = configured
            return self._device

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except Exception:
            self._device = "cpu"

        logger.info("ML device resolved to %s", self._device)
        return self._device

    def torch_threads(self) -> None:
        """
        Cap intra-op threads.

        Torch defaults to every core, which starves the FastAPI threadpool
        that is concurrently serving other requests.
        """
        threads = get_settings().ml_torch_threads
        if threads <= 0:
            return
        try:
            import torch  # type: ignore
            torch.set_num_threads(threads)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Registration & loading
    # ------------------------------------------------------------------
    def register(self, key: str, description: str, extra: str,
                 loader: Callable[[], Any]) -> ModelSlot:
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = ModelSlot(key=key, description=description,
                                 loader=loader, extra=extra)
                self._slots[key] = slot
            return slot

    def get(self, key: str) -> Optional[Any]:
        """
        Return the loaded model, loading it on first use.

        Returns None (and records why) when the dependency is missing or
        the weights cannot be fetched.
        """
        slot = self._slots.get(key)
        if slot is None:
            logger.warning("Unknown model key: %s", key)
            return None

        if slot.model is not None or slot.load_attempted:
            return slot.model

        with slot.load_lock:
            if slot.model is not None or slot.load_attempted:
                return slot.model

            slot.load_attempted = True
            started = time.perf_counter()
            try:
                self.torch_threads()
                slot.model = slot.loader()
                slot.load_seconds = time.perf_counter() - started
                logger.info("Loaded %s in %.1fs", slot.key, slot.load_seconds)
            except ImportError as exc:
                slot.error = (
                    f"missing dependency ({exc}). "
                    f"Install with: pip install -r {slot.extra}"
                )
                logger.warning("%s unavailable - %s", slot.key, slot.error)
            except Exception as exc:
                slot.error = str(exc)
                logger.warning("%s failed to load: %s", slot.key, exc)

        return slot.model

    def infer_lock(self, key: str) -> threading.Lock:
        """Per-model lock; torch modules are not concurrency-safe."""
        slot = self._slots.get(key)
        if slot is None:
            return threading.Lock()
        return slot.infer_lock

    def reset(self, key: Optional[str] = None) -> None:
        """Drop cached models so the next call reloads. Used by tests."""
        with self._lock:
            targets = [key] if key else list(self._slots)
            for name in targets:
                slot = self._slots.get(name)
                if slot is not None:
                    slot.model = None
                    slot.load_attempted = False
                    slot.error = None
                    slot.load_seconds = 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Full picture for the health endpoint."""
        with self._lock:
            return {
                "device": self.device(),
                "models": [slot.status() for slot in self._slots.values()],
            }

    def warmup(self, keys: list[str]) -> None:
        """Load models at startup so the first chunk is not slow."""
        for key in keys:
            self.get(key)


registry = ModelRegistry()
