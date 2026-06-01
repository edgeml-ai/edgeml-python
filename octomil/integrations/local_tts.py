"""Framework-agnostic local TTS cache/prefetch pipeline.

The owning application still decides speaker identity, voice mapping, and
playback. This helper owns the pieces that are common across embedded apps:
client lifecycle, warmup, generated WAV cache, foreground vs speculative
priority, stale-job pruning, and late-play suppression.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

LogCallback = Callable[[str], None]
PlayCallback = Callable[[str], None]
ClientFactory = Callable[[], Any]


@dataclass(frozen=True)
class LocalTTSLine:
    """A line of text and the concrete voice selected by the host app."""

    text: str
    voice: str


@dataclass(frozen=True)
class LocalTTSResult:
    """Result of a cache/play/schedule request."""

    status: str
    path: str
    voice: str


class LocalTTSPipeline:
    """Local TTS pipeline for embedded apps and game runtimes.

    The class is intentionally framework-neutral. A Ren'Py app, a kiosk app,
    or a CLI can all provide the same three inputs:

    - selected voice id,
    - text,
    - callback that plays a completed WAV path.
    """

    def __init__(
        self,
        *,
        model: str,
        cache_dir: str,
        play: PlayCallback,
        client_factory: Optional[ClientFactory] = None,
        policy: str = "private",
        speed: float = 1.0,
        response_format: str = "wav",
        log: Optional[LogCallback] = None,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.play = play
        self.client_factory = client_factory
        self.policy = policy
        self.speed = speed
        self.response_format = response_format
        self.log = log or (lambda _msg: None)

        self.client: Any = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ready = False
        self.error: Optional[str] = None
        self.current_path: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._pending: dict[str, Any] = {}
        self._play_wanted: set[str] = set()
        self._lock = threading.Lock()

        os.makedirs(self.cache_dir, exist_ok=True)

    def start(self) -> None:
        """Start the worker loop and background warmup."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop_thread, name="octomil-local-tts", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Best-effort worker shutdown for tests and short-lived hosts."""
        self.prune_pending(())
        loop = self.loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

    def cache_path(self, line: LocalTTSLine) -> str:
        digest = hashlib.sha256((line.voice + "\x00" + line.text).encode("utf-8")).hexdigest()[:20]
        return os.path.join(self.cache_dir, digest + ".wav")

    def set_current(self, path: Optional[str]) -> None:
        self.current_path = path

    def clear_current(self) -> None:
        """Mark no voiced line current and cancel queued work."""
        self.current_path = None
        self.prune_pending(())

    def play_current(self, line: LocalTTSLine) -> LocalTTSResult:
        """Play a current line from cache or schedule foreground synthesis."""
        path = self.cache_path(line)
        self.current_path = path
        self.prune_pending((path,))
        if os.path.exists(path):
            self._play_path(path, "cache-play", line.voice)
            return LocalTTSResult("cache_hit", path, line.voice)
        if not self.ready:
            self.log("warming, can't synth new line yet: " + line.text[:40])
            return LocalTTSResult("warming", path, line.voice)
        with self._lock:
            in_flight = path in self._pending
            if in_flight:
                self._play_wanted.add(path)
        if in_flight:
            self.log("await in-flight prefetch -> will play voice=%s" % line.voice)
            return LocalTTSResult("pending", path, line.voice)
        self._schedule(line, autoplay=True)
        return LocalTTSResult("scheduled", path, line.voice)

    def prefetch(self, lines: Iterable[LocalTTSLine]) -> None:
        """Schedule speculative cache fills for upcoming lines."""
        if not self.ready:
            return
        for line in lines:
            path = self.cache_path(line)
            if os.path.exists(path):
                self.log("prefetch already cached: %r" % line.text[:40])
                continue
            self._schedule(line, autoplay=False)

    def prune_pending(self, keep_paths: Iterable[str]) -> int:
        """Cancel stale queued work so fast-advance users do not build backlog."""
        keep = set(path for path in keep_paths if path)
        with self._lock:
            stale = [(path, future) for path, future in list(self._pending.items()) if path not in keep]
            for path, future in stale:
                self._pending.pop(path, None)
                self._play_wanted.discard(path)
        cancelled = 0
        for _path, future in stale:
            try:
                future.cancel()
                cancelled += 1
            except Exception:
                pass
        if cancelled:
            self.log("pruned stale synth jobs: %d" % cancelled)
        return cancelled

    def _loop_thread(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            client = self._create_client()
            loop.run_until_complete(self._maybe_await(client.initialize()))
            self._warm_client(loop, client)
            self.client = client
            self.ready = True
            loop.run_forever()
        except Exception as exc:
            self.error = repr(exc)
            self.log("local TTS init failed: " + repr(exc))

    def _create_client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        from octomil import Octomil

        return Octomil.from_env()

    async def _maybe_await(self, value: Any) -> Any:
        if hasattr(value, "__await__"):
            return await value
        return value

    def _warm_client(self, loop: asyncio.AbstractEventLoop, client: Any) -> None:
        try:
            t0 = time.monotonic()
            warm = client.warmup(model=self.model, capability="tts", policy="local_first")
            if hasattr(warm, "__await__"):
                warm = loop.run_until_complete(warm)
            self.log(
                "warmup OK loaded=%s latency_ms=%s elapsed=%.0fms"
                % (
                    getattr(warm, "backend_loaded", "?"),
                    getattr(warm, "latency_ms", "?"),
                    (time.monotonic() - t0) * 1000.0,
                )
            )
        except Exception as exc:
            self.log("warmup failed: " + repr(exc))

    def _schedule(self, line: LocalTTSLine, *, autoplay: bool) -> None:
        if self.client is None or self.loop is None:
            return
        path = self.cache_path(line)
        with self._lock:
            if path in self._pending:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._synthesize_to_cache(line, path, autoplay=autoplay),
                self.loop,
            )
            self._pending[path] = future
        future.add_done_callback(lambda fut: self._on_done(fut, line, path))

    async def _synthesize_to_cache(self, line: LocalTTSLine, path: str, *, autoplay: bool) -> None:
        priority = self._priority(autoplay)
        try:
            response = await self.client.audio.speech.create(
                model=self.model,
                input=line.text,
                voice=line.voice,
                response_format=self.response_format,
                speed=self.speed,
                policy=self.policy,
                priority=priority,
                cache="off",
            )
        except TypeError:
            response = await self.client.audio.speech.create(
                model=self.model,
                input=line.text,
                voice=line.voice,
                response_format=self.response_format,
                speed=self.speed,
                policy=self.policy,
                cache="off",
            )
        data = getattr(response, "audio_bytes", None)
        if not data:
            return
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        if autoplay and self.current_path == path:
            self._play_path(path, "live-play", line.voice)

    def _on_done(self, future: Any, line: LocalTTSLine, path: str) -> None:
        try:
            if future.cancelled():
                self.log("stream bg cancelled voice=%s" % line.voice)
                return
            future.result()
        except Exception as exc:
            self.log("stream bg fail: " + repr(exc))
        finally:
            with self._lock:
                self._pending.pop(path, None)
                wanted = path in self._play_wanted
                self._play_wanted.discard(path)
        if wanted and os.path.exists(path) and self.current_path == path:
            self._play_path(path, "deferred-play", line.voice)

    def _play_path(self, path: str, tag: str, voice: str) -> None:
        try:
            self.play(path)
            self.log("%s OK voice=%s" % (tag, voice))
        except Exception as exc:
            self.log("%s err: %r" % (tag, exc))

    @staticmethod
    def _priority(autoplay: bool) -> Any:
        try:
            from octomil.audio.scheduler import TtsRequestPriority

            return TtsRequestPriority.FOREGROUND if autoplay else TtsRequestPriority.SPECULATIVE
        except Exception:
            return "foreground" if autoplay else "speculative"


__all__ = ["LocalTTSLine", "LocalTTSPipeline", "LocalTTSResult"]
