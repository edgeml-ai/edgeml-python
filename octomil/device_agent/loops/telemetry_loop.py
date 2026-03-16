"""Telemetry loop — batches and uploads telemetry events.

Respects upload policy (connectivity, battery, user consent). Never
blocks inference. Will be fleshed out by the telemetry agent.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TelemetryLoop:
    """Background loop that batches and uploads telemetry events.

    Stub implementation. The telemetry agent will add:
    - Policy-gated upload (WiFi only, battery threshold, user consent)
    - Batching with configurable max_batch_size and flush_interval
    - Exponential backoff on upload failure
    - Storage pressure cleanup
    """

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the telemetry loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the telemetry loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Stub — will be implemented by telemetry agent."""
        logger.info("Telemetry loop started")
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=30.0)
        logger.info("Telemetry loop stopped")
