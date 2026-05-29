"""Runtime smoke test for the generated control-plane client (httpx)."""

from __future__ import annotations

import httpx

from octomil._generated.client import OctomilApiClient


def _client_capturing(captured: list) -> OctomilApiClient:
    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "method": req.method,
                "url": str(req.url),
                "body": req.content.decode() if req.content else None,
            }
        )
        return httpx.Response(200, json={"ok": True})

    return OctomilApiClient(
        "https://api.test", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_get_with_path_param_interpolated():
    cap: list = []
    api = _client_capturing(cap)
    result = api.devices_desired_state(device_id="d1")
    assert result == {"ok": True}
    assert cap[0]["method"] == "GET"
    assert cap[0]["url"] == "https://api.test/api/v1/devices/d1/desired-state"


def test_post_with_json_body():
    cap: list = []
    api = _client_capturing(cap)
    api.telemetry_batch(json={"events": []})
    assert cap[0]["method"] == "POST"
    assert "events" in (cap[0]["body"] or "")


def test_surface_methods_exist():
    api = OctomilApiClient("https://api.test")
    for name in ("devices_desired_state", "telemetry_batch", "federation_join"):
        assert callable(getattr(api, name))
