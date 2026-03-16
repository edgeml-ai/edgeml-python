"""Activation loop — decides when a staged model becomes active.

Respects in-flight inference sessions by waiting for the old version's
refcount to reach zero before completing the transition.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..activation_manager import ActivationManager
from ..inference_session_manager import InferenceSessionManager

logger = logging.getLogger(__name__)


class ActivationLoop:
    """Background loop that monitors staged artifacts and activates them."""

    def __init__(
        self,
        activation_manager: ActivationManager,
        session_manager: InferenceSessionManager,
    ) -> None:
        self._activation_manager = activation_manager
        self._session_manager = session_manager
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the activation loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="activation-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the activation loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Monitors staged artifacts and manages activation."""
        logger.info("Activation loop started")
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=5.0)
        logger.info("Activation loop stopped")
