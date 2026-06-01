"""Local-first telemetry wiring for audio and embeddings."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from octomil.audio.speech import FacadeSpeech, SpeechResponse, SpeechRoute
from octomil.audio.streaming import (
    SpeechAudioChunk,
    SpeechStreamCompleted,
    SpeechStreamStarted,
    TtsStreamingCapability,
)
from octomil.auth import OrgApiKeyAuth, PublishableKeyAuth
from octomil.client import OctomilClient
from octomil.facade import FacadeEmbeddings
from octomil.telemetry import TelemetryReporter


class _SpeechKernel:
    async def synthesize_speech(self, **kwargs):
        return SpeechResponse(
            audio_bytes=b"wav-bytes",
            content_type="audio/wav",
            format=kwargs["response_format"],
            model=kwargs["model"],
            voice=kwargs.get("voice"),
            sample_rate=24_000,
            duration_ms=120,
            latency_ms=4.0,
            route=SpeechRoute(locality="on_device", engine="sherpa-onnx", policy=kwargs.get("policy")),
        )


class _StreamKernel:
    async def synthesize_speech_stream(self, **kwargs):
        async def _events():
            cap = TtsStreamingCapability.final_only()
            yield SpeechStreamStarted(
                model=kwargs["model"],
                voice=kwargs.get("voice"),
                sample_rate=24_000,
                channels=1,
                sample_format="pcm_s16le",
                streaming_capability=cap,
                locality="on_device",
                engine="sherpa-onnx",
            )
            yield SpeechAudioChunk(data=b"\x00\x00", sample_index=1, timestamp_ms=0)
            yield SpeechStreamCompleted(
                duration_ms=1,
                total_samples=1,
                sample_rate=24_000,
                channels=1,
                sample_format="pcm_s16le",
                streaming_capability=cap,
                setup_ms=1.0,
                engine_first_chunk_ms=1.0,
                e2e_first_chunk_ms=1.0,
                total_latency_ms=2.0,
                observed_chunks=1,
                capability_verified=True,
            )

        return _events()


def test_telemetry_reporter_exposes_track_event_bridge() -> None:
    reporter = cast(Any, object.__new__(TelemetryReporter))
    setattr(reporter, "_enqueue", MagicMock())

    TelemetryReporter.track(reporter, "local.event", {"safe": True})
    TelemetryReporter.track_event(reporter, "local.event.alias", {"safe": False})

    assert reporter._enqueue.call_args_list[0].kwargs == {
        "name": "local.event",
        "attributes": {"safe": True},
    }
    assert reporter._enqueue.call_args_list[1].kwargs == {
        "name": "local.event.alias",
        "attributes": {"safe": False},
    }


def test_publishable_key_initializes_telemetry_reporter() -> None:
    with (
        patch("octomil.client.RolloutsAPI", create=True),
        patch("octomil.client.ModelRegistry", create=True),
        patch("octomil.client._ApiClient", create=True),
        patch("octomil.telemetry.TelemetryReporter") as reporter_cls,
    ):
        reporter = MagicMock()
        reporter_cls.return_value = reporter
        client = OctomilClient(auth=PublishableKeyAuth(api_key="oct_pub_test_abc123"))

    assert client._reporter is reporter
    reporter_cls.assert_called_once()
    assert reporter_cls.call_args.kwargs["api_key"] == "oct_pub_test_abc123"


@pytest.mark.asyncio
async def test_speech_create_emits_local_usage_without_input_text() -> None:
    reporter = MagicMock()
    speech = FacadeSpeech(_SpeechKernel(), telemetry_reporter=reporter)

    await speech.create(model="kokoro-82m", input="secret dialogue", voice="af_bella", cache="off")

    reporter.track_event.assert_called_once()
    event_name, attrs = reporter.track_event.call_args.args
    assert event_name == "tts.create.completed"
    assert attrs["model.id"] == "kokoro-82m"
    assert attrs["locality"] == "on_device"
    assert attrs["cache.hit"] is False
    assert "secret dialogue" not in repr(attrs)
    assert "input" not in attrs


@pytest.mark.asyncio
async def test_speech_stream_attaches_telemetry_sink() -> None:
    reporter = MagicMock()
    speech = FacadeSpeech(_StreamKernel(), telemetry_reporter=reporter)

    stream = speech.stream(model="kokoro-82m", input="secret dialogue", voice="af_bella")
    await stream.collect()

    event_names = [call.args[0] for call in reporter.track_event.call_args_list]
    assert "tts.stream.started" in event_names
    assert "tts.stream.completed" in event_names
    for call in reporter.track_event.call_args_list:
        assert "secret dialogue" not in repr(call.args[1])


@pytest.mark.asyncio
async def test_local_embeddings_emit_usage_without_input_text() -> None:
    reporter = MagicMock()

    class _EmbeddingKernel:
        async def create_embeddings(self, inputs, **kwargs):
            return SimpleNamespace(
                embeddings=[[0.1, 0.2, 0.3]],
                model=kwargs["model"],
                usage={"prompt_tokens": 3, "total_tokens": 3},
                route=SimpleNamespace(locality="on_device"),
            )

    client = SimpleNamespace(_auth=OrgApiKeyAuth(api_key="test", org_id="org"), _reporter=reporter)
    embeddings = FacadeEmbeddings(cast(Any, client), kernel=_EmbeddingKernel())

    await embeddings.create(model="embed-local", input="private text", policy="local_only")

    reporter.track_event.assert_called_once()
    event_name, attrs = reporter.track_event.call_args.args
    assert event_name == "embeddings.create.completed"
    assert attrs["model.id"] == "embed-local"
    assert attrs["input.count"] == 1
    assert attrs["embedding.dimensions"] == 3
    assert "private text" not in repr(attrs)
