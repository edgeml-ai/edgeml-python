"""Tests for keyless local facade behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from octomil.auth import NoAuth
from octomil.errors import OctomilError, OctomilErrorCode
from octomil.facade import Octomil


class TestKeylessOctomil:
    def test_no_arg_constructor_uses_keyless_local_mode(self) -> None:
        client = Octomil()

        assert isinstance(client._auth, NoAuth)
        assert client.planner_enabled is False

    def test_from_env_without_server_key_uses_keyless_local_mode(self, monkeypatch) -> None:
        monkeypatch.delenv("OCTOMIL_SERVER_KEY", raising=False)
        monkeypatch.delenv("OCTOMIL_API_KEY", raising=False)
        monkeypatch.setenv("OCTOMIL_ORG_ID", "org_public_id")

        client = Octomil.from_env()

        assert isinstance(client._auth, NoAuth)
        assert client.planner_enabled is False

    def test_public_local_constructor_is_removed(self) -> None:
        assert not hasattr(Octomil, "local")

    @pytest.mark.asyncio
    async def test_keyless_exposes_full_audio_surface(self) -> None:
        from octomil.audio import FacadeAudio, FacadeSpeech

        with (
            patch("octomil.client.RolloutsAPI", create=True),
            patch("octomil.client.ModelRegistry", create=True),
            patch("octomil.client._ApiClient", create=True),
        ):
            client = Octomil()
            await client.initialize()

        assert isinstance(client.audio, FacadeAudio)
        assert isinstance(client.audio.speech, FacadeSpeech)
        assert callable(client.audio.speech.create)

    @pytest.mark.asyncio
    async def test_keyless_rejects_explicit_cloud_policy(self) -> None:
        with (
            patch("octomil.client.RolloutsAPI", create=True),
            patch("octomil.client.ModelRegistry", create=True),
            patch("octomil.client._ApiClient", create=True),
        ):
            client = Octomil()
            await client.initialize()

        with pytest.raises(OctomilError) as exc:
            await client.audio.speech.create(
                model="kokoro-82m",
                input="hello",
                policy="cloud_only",
                cache="off",
            )

        assert exc.value.code == OctomilErrorCode.INVALID_API_KEY
        assert "requires Octomil credentials" in str(exc.value)
