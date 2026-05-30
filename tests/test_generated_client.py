"""Runtime smoke test for the generated control-plane client (httpx)."""

from __future__ import annotations

import inspect

import httpx
import pydantic
import pytest

from octomil._generated.client import OctomilApiClient


def _client(captured: list, payload: dict) -> OctomilApiClient:
    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "method": req.method,
                "url": str(req.url),
                "body": req.content.decode() if req.content else None,
            }
        )
        return httpx.Response(200, json=payload)

    return OctomilApiClient(
        "https://api.test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_request_building_path_and_body():
    """A dict-returning (untyped-response) method: verb, path interpolation,
    and JSON body are built correctly."""
    cap: list = []
    api = _client(cap, {"ok": True})
    # devices.observed_state — POST /api/v1/devices/{device_id}/observed-state
    result = api.devices_observed_state(device_id="d1", json={"battery_pct": 80})
    assert result == {"ok": True}  # Any return == parsed JSON
    assert cap[0]["method"] == "POST"
    assert cap[0]["url"] == "https://api.test/api/v1/devices/d1/observed-state"
    assert "battery_pct" in (cap[0]["body"] or "")


def test_response_is_typed_to_the_model():
    """Methods with a contract response schema are typed to the generated
    pydantic model, and model_validate is actually applied."""
    # return annotation is the model (string under `from __future__ annotations`)
    ann = inspect.signature(OctomilApiClient.devices_desired_state).return_annotation
    assert ann == "DesiredState"
    # model_validate runs: a payload that doesn't satisfy the model raises
    cap: list = []
    api = _client(cap, {"not": "a desired state"})
    with pytest.raises(pydantic.ValidationError):
        api.devices_desired_state(device_id="d1")


def test_surface_methods_exist():
    api = OctomilApiClient("https://api.test")
    for name in ("devices_desired_state", "telemetry_batch", "federation_join"):
        assert callable(getattr(api, name))
