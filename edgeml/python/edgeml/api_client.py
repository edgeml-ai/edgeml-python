from __future__ import annotations

from typing import Any, Callable, Optional

import httpx


class EdgeMLClientError(RuntimeError):
    pass


class _ApiClient:
    def __init__(
        self,
        auth_token_provider: Callable[[], str],
        api_base: str,
        timeout: float = 20.0,
    ):
        self.auth_token_provider = auth_token_provider
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        token = self.auth_token_provider()
        if not token:
            raise EdgeMLClientError("auth_token_provider returned an empty token")
        return {"Authorization": f"Bearer {token}"}

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def post(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.api_base}{path}", json=payload or {}, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def put(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.put(f"{self.api_base}{path}", json=payload or {}, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json() if res.text else {}

    def patch(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.patch(f"{self.api_base}{path}", json=payload or {}, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json() if res.text else {}

    def delete(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.delete(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json() if res.text else {}

    def get_bytes(self, path: str, params: Optional[dict[str, Any]] = None) -> bytes:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.api_base}{path}", params=params, headers=self._headers())
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.content

    def post_bytes(self, path: str, data: bytes) -> Any:
        """POST raw bytes (``application/octet-stream``)."""
        headers = {**self._headers(), "Content-Type": "application/octet-stream"}
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.api_base}{path}", content=data, headers=headers)
        if res.status_code >= 400:
            raise EdgeMLClientError(res.text)
        return res.json()

    def report_inference_event(self, payload: dict[str, Any]) -> Any:
        """Report a streaming inference event to ``POST /inference/events``."""
        return self.post("/inference/events", payload)

    # ------------------------------------------------------------------
    # SecAgg endpoints
    # ------------------------------------------------------------------

    def secagg_get_session(self, round_id: str, device_id: str) -> Any:
        """Fetch the SecAgg session config for a round."""
        return self.get(
            f"/training/rounds/{round_id}/secagg/session",
            params={"device_id": device_id},
        )

    def secagg_submit_shares(
        self, round_id: str, device_id: str, shares_data: bytes
    ) -> Any:
        """Upload this client's Shamir key-shares for the round."""
        return self.post_bytes(
            f"/training/rounds/{round_id}/secagg/shares?device_id={device_id}",
            shares_data,
        )

    def secagg_submit_masked_update(
        self, round_id: str, device_id: str, masked_data: bytes
    ) -> Any:
        """Upload the masked model update."""
        return self.post_bytes(
            f"/training/rounds/{round_id}/secagg/masked?device_id={device_id}",
            masked_data,
        )

    def secagg_submit_unmask_share(
        self, round_id: str, device_id: str, peer_id: str, share_data: bytes
    ) -> Any:
        """Reveal a Shamir share for a dropped-out peer."""
        return self.post_bytes(
            f"/training/rounds/{round_id}/secagg/unmask"
            f"?device_id={device_id}&peer_id={peer_id}",
            share_data,
        )

