"""Control plane -- device registration and heartbeat (SDK Facade Contract namespace)."""

from __future__ import annotations

import logging
import platform
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from .device_info import DeviceInfo
from .python.octomil.api_client import _ApiClient

logger = logging.getLogger(__name__)


@dataclass
class DeviceRegistration:
    """Response from device registration."""

    id: str
    device_identifier: str
    org_id: str
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatResponse:
    """Response from heartbeat."""

    status: str = "ok"
    server_time: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OctomilControl:
    """Device registration and heartbeat management.

    Matches the ``control`` namespace in SDK_FACADE_CONTRACT.md:
    - refresh() -> void
    - register(deviceId?) -> DeviceRegistration
    - heartbeat() -> HeartbeatResponse
    """

    def __init__(self, api: _ApiClient, org_id: str) -> None:
        self._api = api
        self._org_id = org_id
        self._device_info = DeviceInfo()
        self._server_device_id: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop: threading.Event = threading.Event()

    def refresh(self) -> None:
        """Fetch latest assignments and rollout state from server."""
        if not self._server_device_id:
            return
        self._api.get(f"/devices/{self._server_device_id}/assignments")

    def register(self, device_id: Optional[str] = None) -> DeviceRegistration:
        """Register device with server. Returns DeviceRegistration."""
        effective_device_id = device_id or self._device_info.device_id

        payload = self._device_info.to_registration_dict()
        payload["device_identifier"] = effective_device_id
        payload["org_id"] = self._org_id
        payload["sdk_version"] = _get_sdk_version()

        data = self._api.post("/devices/register", payload)

        self._server_device_id = data.get("id", "")
        return DeviceRegistration(
            id=data.get("id", ""),
            device_identifier=effective_device_id,
            org_id=self._org_id,
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
        )

    def heartbeat(self) -> HeartbeatResponse:
        """Send device heartbeat to server."""
        if not self._server_device_id:
            raise RuntimeError("Device not registered. Call register() first.")

        payload: dict[str, Any] = {
            "sdk_version": _get_sdk_version(),
            "os_version": platform.platform(),
            "platform": "python",
        }

        # Merge runtime metadata (battery, network) -- best-effort
        try:
            payload["metadata"] = self._device_info.update_metadata()
        except Exception:
            pass

        data = self._api.post(
            f"/devices/{self._server_device_id}/heartbeat",
            payload,
        )

        return HeartbeatResponse(
            status=data.get("status", "ok"),
            server_time=data.get("server_time"),
            metadata=data.get("metadata", {}),
        )

    def start_heartbeat(self, interval_seconds: float = 300.0) -> None:
        """Start automatic heartbeat in background thread."""
        self.stop_heartbeat()
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval_seconds,),
            daemon=True,
            name="octomil-heartbeat",
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop automatic heartbeat."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._heartbeat_stop.wait(timeout=interval):
            try:
                self.heartbeat()
            except Exception:
                logger.debug("Heartbeat failed", exc_info=True)


def _get_sdk_version() -> str:
    try:
        from octomil import __version__

        return __version__
    except ImportError:
        return "0.0.0"
