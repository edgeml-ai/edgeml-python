"""Artifact loop — downloads, resumes, verifies, and stages artifacts.

Polls the operations table for download/verify tasks and drives the
ArtifactDownloader and ArtifactVerifier through the artifact lifecycle.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..artifact_downloader import ArtifactDownloader
from ..artifact_verifier import ArtifactVerifier
from ..operation_scheduler import OperationScheduler

logger = logging.getLogger(__name__)


class ArtifactLoop:
    """Background loop that processes artifact download and verification operations."""

    def __init__(
        self,
        downloader: ArtifactDownloader,
        verifier: ArtifactVerifier,
        scheduler: OperationScheduler,
    ) -> None:
        self._downloader = downloader
        self._verifier = verifier
        self._scheduler = scheduler
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the artifact loop in a background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="artifact-loop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the artifact loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """Loop body. Polls for pending artifact operations."""
        logger.info("Artifact loop started")
        while not self._stop_event.is_set():
            # Recover any expired leases
            self._scheduler.recover_expired_leases()
            # Poll interval
            self._stop_event.wait(timeout=5.0)
        logger.info("Artifact loop stopped")
