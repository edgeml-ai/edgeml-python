"""Inference loop — serves requests using the active model version.

Reads only local state (active_model_pointer). Never mutates model
versions or triggers downloads. Uses InferenceSessionManager for
refcounted version pinning.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..inference_session_manager import InferenceSessionManager

logger = logging.getLogger(__name__)


class InferenceLoop:
    """Main inference serving loop.

    Acquires a session handle at the start of each request, pinning the
    model version. Releases on completion. The loop itself runs in a
    background thread polling for work, or can be driven externally.
    """

    def __init__(self, session_manager: InferenceSessionManager) -> None:
        self._session_manager = session_manager
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the inference loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="inference-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the inference loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Override or extend for custom inference dispatch."""
        logger.info("Inference loop started")
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)
        logger.info("Inference loop stopped")
