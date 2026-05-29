"""Generated control-plane API client — DO NOT EDIT.

Source: octomil-contracts ci/client_sdk_surface.yaml + methods/*.yaml
Generator: tools/sdkgen/gen_client_py.py
"""

from __future__ import annotations

from typing import Any

import httpx


class OctomilApiClient:
    """Typed control-plane API client (generated). Thin httpx wrapper; one
    method per HTTP client-surface operation. Request params are typed; the
    parsed JSON response is returned (response-model typing is a later layer)."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = client or httpx.Client()
        self._headers = headers or {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> Any:
        resp = self._client.request(
            method,
            self._base + path,
            params=params,
            json=json,
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def artifacts_download_urls(self, *, artifact_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """artifacts.download_urls — POST /api/v1/artifacts/{artifact_id}/download-urls"""
        return self._request("POST", f"/api/v1/artifacts/{artifact_id}/download-urls", params=params, json=json)

    def artifacts_manifest(self, *, artifact_id: str, params: dict[str, Any] | None = None) -> Any:
        """artifacts.manifest — GET /api/v1/artifacts/{artifact_id}/manifest"""
        return self._request("GET", f"/api/v1/artifacts/{artifact_id}/manifest", params=params)

    def audio_speech_create(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """audio.speech.create — POST /v1/audio/speech"""
        return self._request("POST", "/v1/audio/speech", params=params, json=json)

    def chat_threads_create(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """chat.threads.create — POST /api/v1/chat/threads"""
        return self._request("POST", "/api/v1/chat/threads", params=params, json=json)

    def chat_threads_get(self, *, thread_id: str, params: dict[str, Any] | None = None) -> Any:
        """chat.threads.get — GET /api/v1/chat/threads/{thread_id}"""
        return self._request("GET", f"/api/v1/chat/threads/{thread_id}", params=params)

    def chat_threads_list(self, *, params: dict[str, Any] | None = None) -> Any:
        """chat.threads.list — GET /api/v1/chat/threads"""
        return self._request("GET", "/api/v1/chat/threads", params=params)

    def chat_turn(self, *, thread_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """chat.turn — POST /api/v1/chat/threads/{thread_id}/turns"""
        return self._request("POST", f"/api/v1/chat/threads/{thread_id}/turns", params=params, json=json)

    def devices_desired_state(self, *, device_id: str, params: dict[str, Any] | None = None) -> Any:
        """devices.desired_state — GET /api/v1/devices/{device_id}/desired-state"""
        return self._request("GET", f"/api/v1/devices/{device_id}/desired-state", params=params)

    def devices_observed_state(self, *, device_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """devices.observed_state — POST /api/v1/devices/{device_id}/observed-state"""
        return self._request("POST", f"/api/v1/devices/{device_id}/observed-state", params=params, json=json)

    def devices_sync(self, *, device_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """devices.sync — POST /api/v1/devices/{device_id}/sync"""
        return self._request("POST", f"/api/v1/devices/{device_id}/sync", params=params, json=json)

    def federation_heartbeat(self, *, round_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """federation.heartbeat — POST /api/v1/federation/rounds/{round_id}/heartbeat"""
        return self._request("POST", f"/api/v1/federation/rounds/{round_id}/heartbeat", params=params, json=json)

    def federation_join(self, *, round_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """federation.join — POST /api/v1/federation/rounds/{round_id}/join"""
        return self._request("POST", f"/api/v1/federation/rounds/{round_id}/join", params=params, json=json)

    def federation_offers(self, *, params: dict[str, Any] | None = None) -> Any:
        """federation.offers — GET /api/v1/federation/rounds/offers"""
        return self._request("GET", "/api/v1/federation/rounds/offers", params=params)

    def federation_plan(self, *, plan_id: str, params: dict[str, Any] | None = None) -> Any:
        """federation.plan — GET /api/v1/federation/plans/{plan_id}"""
        return self._request("GET", f"/api/v1/federation/plans/{plan_id}", params=params)

    def federation_upload_complete(self, *, round_id: str, upload_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """federation.upload_complete — POST /api/v1/federation/rounds/{round_id}/updates/{upload_id}/complete"""
        return self._request("POST", f"/api/v1/federation/rounds/{round_id}/updates/{upload_id}/complete", params=params, json=json)

    def federation_upload_initiate(self, *, round_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """federation.upload_initiate — POST /api/v1/federation/rounds/{round_id}/updates/initiate"""
        return self._request("POST", f"/api/v1/federation/rounds/{round_id}/updates/initiate", params=params, json=json)

    def monitoring_alerts_delete(self, *, rule_id: str, params: dict[str, Any] | None = None) -> Any:
        """monitoring.alerts.delete — DELETE /api/v1/monitoring/alerts/{rule_id}"""
        return self._request("DELETE", f"/api/v1/monitoring/alerts/{rule_id}", params=params)

    def monitoring_alerts_get(self, *, rule_id: str, params: dict[str, Any] | None = None) -> Any:
        """monitoring.alerts.get — GET /api/v1/monitoring/alerts/{rule_id}"""
        return self._request("GET", f"/api/v1/monitoring/alerts/{rule_id}", params=params)

    def monitoring_alerts_update(self, *, rule_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """monitoring.alerts.update — PATCH /api/v1/monitoring/alerts/{rule_id}"""
        return self._request("PATCH", f"/api/v1/monitoring/alerts/{rule_id}", params=params, json=json)

    def responses_create(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """responses.create — POST /v1/responses"""
        return self._request("POST", "/v1/responses", params=params, json=json)

    def responses_stream(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """responses.stream — POST /v1/responses"""
        return self._request("POST", "/v1/responses", params=params, json=json)

    def settings_billing_create_checkout_session(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """settings.billing.create_checkout_session — POST /api/v1/settings/billing/checkout"""
        return self._request("POST", "/api/v1/settings/billing/checkout", params=params, json=json)

    def settings_billing_create_portal_session(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """settings.billing.create_portal_session — POST /api/v1/settings/billing/portal"""
        return self._request("POST", "/api/v1/settings/billing/portal", params=params, json=json)

    def settings_billing_update(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """settings.billing.update — PATCH /api/v1/settings/billing"""
        return self._request("PATCH", "/api/v1/settings/billing", params=params, json=json)

    def settings_integrations_delete(self, *, integration_id: str, params: dict[str, Any] | None = None) -> Any:
        """settings.integrations.delete — DELETE /api/v1/settings/integrations/{integration_id}"""
        return self._request("DELETE", f"/api/v1/settings/integrations/{integration_id}", params=params)

    def settings_integrations_get(self, *, integration_id: str, params: dict[str, Any] | None = None) -> Any:
        """settings.integrations.get — GET /api/v1/settings/integrations/{integration_id}"""
        return self._request("GET", f"/api/v1/settings/integrations/{integration_id}", params=params)

    def settings_integrations_update(self, *, integration_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """settings.integrations.update — PATCH /api/v1/settings/integrations/{integration_id}"""
        return self._request("PATCH", f"/api/v1/settings/integrations/{integration_id}", params=params, json=json)

    def settings_integrations_validate(self, *, integration_id: str, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """settings.integrations.validate — POST /api/v1/settings/integrations/{integration_id}/validate"""
        return self._request("POST", f"/api/v1/settings/integrations/{integration_id}/validate", params=params, json=json)

    def settings_usage_limits_get(self, *, params: dict[str, Any] | None = None) -> Any:
        """settings.usage_limits.get — GET /api/v1/settings/usage-limits"""
        return self._request("GET", "/api/v1/settings/usage-limits", params=params)

    def settings_usage_limits_update(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """settings.usage_limits.update — PUT /api/v1/settings/usage-limits"""
        return self._request("PUT", "/api/v1/settings/usage-limits", params=params, json=json)

    def telemetry_batch(self, *, params: dict[str, Any] | None = None, json: Any | None = None) -> Any:
        """telemetry.batch — POST /api/v1/telemetry/batches"""
        return self._request("POST", "/api/v1/telemetry/batches", params=params, json=json)

