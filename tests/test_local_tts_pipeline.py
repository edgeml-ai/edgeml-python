from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from octomil.integrations.local_tts import LocalTTSLine, LocalTTSPipeline


@dataclass
class _Warmup:
    backend_loaded: bool = True
    latency_ms: float = 1.0


@dataclass
class _SpeechResponse:
    audio_bytes: bytes


class _FakeSpeech:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.started: list[str] = []
        self.cancelled: list[str] = []

    async def create(self, **kwargs):
        text = kwargs["input"]
        self.started.append(text)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        return _SpeechResponse(b"RIFFfake-wav-" + text.encode("utf-8"))


class _FakeAudio:
    def __init__(self, speech: _FakeSpeech) -> None:
        self.speech = speech


class _FakeClient:
    def __init__(self, speech: _FakeSpeech) -> None:
        self.audio = _FakeAudio(speech)

    async def initialize(self):
        return None

    def warmup(self, **_kwargs):
        return _Warmup()


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_cache_hit_plays_without_client(tmp_path):
    played: list[str] = []
    pipeline = LocalTTSPipeline(
        model="kokoro-82m",
        cache_dir=str(tmp_path),
        play=played.append,
    )
    line = LocalTTSLine("hello", "af_bella")
    path = pipeline.cache_path(line)
    tmp_path.joinpath(path.split("/")[-1]).write_bytes(b"wav")

    result = pipeline.play_current(line)

    assert result.status == "cache_hit"
    assert played == [path]


def test_rapid_advance_prunes_stale_pending_jobs(tmp_path):
    played: list[str] = []
    speech = _FakeSpeech(delay=0.2)
    pipeline = LocalTTSPipeline(
        model="kokoro-82m",
        cache_dir=str(tmp_path),
        play=played.append,
        client_factory=lambda: _FakeClient(speech),
    )
    pipeline.start()
    _wait_until(lambda: pipeline.ready)

    first = LocalTTSLine("line one", "af_bella")
    second = LocalTTSLine("line fifty", "af_sarah")
    assert pipeline.play_current(first).status == "scheduled"
    _wait_until(lambda: bool(speech.started))
    assert pipeline.play_current(second).status == "scheduled"

    second_path = pipeline.cache_path(second)
    _wait_until(lambda: second_path in played)

    assert first.text in speech.cancelled
    assert played == [second_path]
    pipeline.stop()


def test_prefetch_caught_by_current_line_plays_when_done(tmp_path):
    played: list[str] = []
    speech = _FakeSpeech(delay=0.03)
    pipeline = LocalTTSPipeline(
        model="kokoro-82m",
        cache_dir=str(tmp_path),
        play=played.append,
        client_factory=lambda: _FakeClient(speech),
    )
    pipeline.start()
    _wait_until(lambda: pipeline.ready)

    line = LocalTTSLine("prefetched line", "af_nova")
    pipeline.prefetch([line])
    _wait_until(lambda: bool(speech.started))
    assert pipeline.play_current(line).status == "pending"

    path = pipeline.cache_path(line)
    _wait_until(lambda: path in played)

    assert played == [path]
    pipeline.stop()
