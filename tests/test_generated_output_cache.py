"""Tests for the generated-output cache used by local TTS."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from octomil.audio.speech import FacadeSpeech, SpeechResponse, SpeechRoute
from octomil.runtime.lifecycle.output_cache import GeneratedOutputCache, derive_output_key


class _SpeechKernel:
    def __init__(self) -> None:
        self.calls = 0

    async def synthesize_speech(self, **kwargs):
        self.calls += 1
        return SpeechResponse(
            audio_bytes=f"audio-{self.calls}".encode("ascii"),
            content_type="audio/wav",
            format=kwargs["response_format"],
            model=kwargs["model"],
            provider=None,
            voice=kwargs.get("voice"),
            sample_rate=24_000,
            duration_ms=120,
            latency_ms=3.5,
            route=SpeechRoute(locality="on_device", engine="sherpa-onnx", policy=kwargs.get("policy")),
        )


class _UnavailableSpeechKernel:
    async def synthesize_speech(self, **kwargs):
        raise AssertionError("engine should not be touched for generated-output cache hits")


def test_cache_round_trip_and_hex_key_validation(tmp_path: Path) -> None:
    cache = GeneratedOutputCache(root=tmp_path, max_bytes=1024)
    key = derive_output_key("audio.speech", model="kokoro-82m", payload={"input": "hi"})

    cache.put("audio.speech", key, b"wav-bytes", {"format": "wav"})
    hit = cache.get("audio.speech", key)

    assert hit is not None
    assert hit.data == b"wav-bytes"
    assert hit.metadata == {"format": "wav"}

    with pytest.raises(ValueError, match="64-character hex digest"):
        cache.get("audio.speech", "not-a-cache-key")


def test_cache_evicts_oldest_entry_to_global_budget(tmp_path: Path) -> None:
    cache = GeneratedOutputCache(root=tmp_path, max_bytes=64)
    first = derive_output_key("audio.speech", model="kokoro-82m", payload={"input": "first"})
    second = derive_output_key("audio.speech", model="kokoro-82m", payload={"input": "second"})

    cache.put("audio.speech", first, b"a" * 32)
    time.sleep(0.01)
    cache.put("embeddings", second, b"b" * 32)

    assert cache.get("audio.speech", first) is None
    assert cache.get("embeddings", second) is not None


@pytest.mark.asyncio
async def test_speech_create_uses_generated_output_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OCTOMIL_OUTPUT_CACHE_DIR", str(tmp_path))
    kernel = _SpeechKernel()
    speech = FacadeSpeech(kernel)

    first = await speech.create(model="kokoro-82m", input="hello", voice="af_bella")
    second = await speech.create(model="kokoro-82m", input="hello", voice="af_bella")
    refreshed = await speech.create(model="kokoro-82m", input="hello", voice="af_bella", cache="refresh")

    assert first.audio_bytes == b"audio-1"
    assert second.audio_bytes == b"audio-1"
    assert second.latency_ms == 0.0
    assert refreshed.audio_bytes == b"audio-2"
    assert kernel.calls == 2


@pytest.mark.asyncio
async def test_speech_cache_hit_returns_before_engine_ready(monkeypatch, tmp_path: Path) -> None:
    """Cached generated audio must be playable while warmup/model load is still cold."""
    monkeypatch.setenv("OCTOMIL_OUTPUT_CACHE_DIR", str(tmp_path))
    key = derive_output_key(
        "audio.speech",
        model="kokoro-82m",
        payload={
            "input": "hello",
            "voice": "af_bella",
            "speaker": None,
            "response_format": "wav",
            "speed": 1.0,
            "app": None,
            "text_normalization": "auto",
        },
    )
    GeneratedOutputCache(root=tmp_path / "outputs").put(
        "audio.speech",
        key,
        b"cached-wav",
        {
            "content_type": "audio/wav",
            "format": "wav",
            "model": "kokoro-82m",
            "voice": "af_bella",
            "sample_rate": 24_000,
            "duration_ms": 120,
            "locality": "on_device",
            "engine": "sherpa-onnx",
            "policy": "local_first",
        },
    )
    speech = FacadeSpeech(_UnavailableSpeechKernel())

    response = await speech.create(model="kokoro-82m", input="hello", voice="af_bella")

    assert response.audio_bytes == b"cached-wav"
    assert response.latency_ms == 0.0
    assert response.route.engine == "sherpa-onnx"


@pytest.mark.asyncio
async def test_speech_cache_write_failure_is_fail_open(monkeypatch, tmp_path: Path) -> None:
    from octomil.runtime.lifecycle.output_cache import GeneratedOutputCache

    monkeypatch.setenv("OCTOMIL_OUTPUT_CACHE_DIR", str(tmp_path))

    def _raise(*args, **kwargs) -> None:
        raise OSError("read-only cache dir")

    monkeypatch.setattr(GeneratedOutputCache, "put", _raise)
    kernel = _SpeechKernel()
    speech = FacadeSpeech(kernel)

    response = await speech.create(model="kokoro-82m", input="hello")

    assert response.audio_bytes == b"audio-1"
    assert kernel.calls == 1


@pytest.mark.asyncio
async def test_speech_cache_key_ignores_routing_policy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OCTOMIL_OUTPUT_CACHE_DIR", str(tmp_path))
    kernel = _SpeechKernel()
    speech = FacadeSpeech(kernel)

    first = await speech.create(model="kokoro-82m", input="hello", policy="local_only")
    second = await speech.create(model="kokoro-82m", input="hello", policy="private")

    assert first.audio_bytes == b"audio-1"
    assert second.audio_bytes == b"audio-1"
    assert kernel.calls == 1
