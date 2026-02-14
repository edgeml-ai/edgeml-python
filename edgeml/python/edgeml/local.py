"""Local-first model instrumentation for EdgeML.

Provides ``model()`` decorator and ``connect()`` for optional server reporting.
Works fully offline by default — no API key, no server, no registration needed.
"""

from __future__ import annotations

import inspect
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, List, Optional


@dataclass
class _Connection:
    api_key: str
    api_base: str
    org_id: str


@dataclass
class ModelRunResult:
    model_name: str
    format: Optional[str]
    version: Optional[str]
    is_streaming: bool
    total_duration_ms: float

    # Streaming-only
    ttfc_ms: Optional[float] = None
    chunk_count: Optional[int] = None
    throughput: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    avg_chunk_latency_ms: Optional[float] = None

    # Non-streaming only
    latency_ms: Optional[float] = None

    def summary_line(self) -> str:
        parts = [f"[edgeml] {self.model_name}"]
        if self.format:
            parts.append(self.format)
        if self.version:
            parts.append(self.version)

        if self.is_streaming:
            if self.chunk_count is not None:
                parts.append(f"| {self.chunk_count} tok")
            if self.ttfc_ms is not None:
                parts.append(f"| TTFC {self.ttfc_ms:.0f}ms")
            if self.throughput is not None:
                parts.append(f"| {self.throughput:.1f} tok/s")
            if self.p99_latency_ms is not None:
                parts.append(f"| p99 {self.p99_latency_ms:.0f}ms")
        else:
            if self.latency_ms is not None:
                parts.append(f"| {self.latency_ms:.0f}ms")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_name": self.model_name,
            "format": self.format,
            "version": self.version,
            "is_streaming": self.is_streaming,
            "total_duration_ms": self.total_duration_ms,
        }
        if self.is_streaming:
            d["ttfc_ms"] = self.ttfc_ms
            d["chunk_count"] = self.chunk_count
            d["throughput"] = self.throughput
            d["p99_latency_ms"] = self.p99_latency_ms
            d["avg_chunk_latency_ms"] = self.avg_chunk_latency_ms
        else:
            d["latency_ms"] = self.latency_ms
        return d


_connection: Optional[_Connection] = None


class TrackedModel:
    """Wraps an inference function with timing instrumentation."""

    def __init__(self, name: str, format: Optional[str] = None, version: Optional[str] = None) -> None:
        self.name = name
        self.format = format
        self.version = version
        self.last_result: Optional[ModelRunResult] = None
        self._history: List[ModelRunResult] = []

    def __call__(self, fn: Callable) -> Callable:
        if inspect.isgeneratorfunction(fn):
            wrapper = self._wrap_generator(fn)
        else:
            wrapper = self._wrap_regular(fn)
        wrapper._tracked_model = self  # type: ignore[attr-defined]
        return wrapper

    def stream(self, fn: Callable, *args: Any, **kwargs: Any) -> Generator:
        return self._instrumented_generator(fn, args, kwargs)

    def run(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return self._instrumented_call(fn, args, kwargs)

    def metrics(self) -> List[ModelRunResult]:
        return list(self._history)

    # -- internal --------------------------------------------------------

    def _wrap_generator(self, fn: Callable) -> Callable:
        tracked = self

        def wrapper(*args: Any, **kwargs: Any) -> Generator:
            return tracked._instrumented_generator(fn, args, kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    def _wrap_regular(self, fn: Callable) -> Callable:
        tracked = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return tracked._instrumented_call(fn, args, kwargs)

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    def _instrumented_generator(self, fn: Callable, args: tuple, kwargs: dict) -> Generator:
        latencies: list[float] = []
        chunk_count = 0
        first_chunk_time: Optional[float] = None
        start = time.monotonic()
        prev = start
        try:
            for chunk in fn(*args, **kwargs):
                now = time.monotonic()
                chunk_count += 1
                if first_chunk_time is None:
                    first_chunk_time = now
                latencies.append((now - prev) * 1000)
                prev = now
                yield chunk
        finally:
            end = time.monotonic()
            total_ms = (end - start) * 1000
            ttfc_ms = ((first_chunk_time - start) * 1000) if first_chunk_time else None
            avg_lat = (sum(latencies) / len(latencies)) if latencies else None
            p99 = self._p99(latencies) if latencies else None
            total_sec = total_ms / 1000
            tps = (chunk_count / total_sec) if total_sec > 0 else None
            result = ModelRunResult(
                model_name=self.name,
                format=self.format,
                version=self.version,
                is_streaming=True,
                total_duration_ms=total_ms,
                ttfc_ms=ttfc_ms,
                chunk_count=chunk_count,
                throughput=tps,
                p99_latency_ms=p99,
                avg_chunk_latency_ms=avg_lat,
            )
            self._record(result)

    def _instrumented_call(self, fn: Callable, args: tuple, kwargs: dict) -> Any:
        start = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            end = time.monotonic()
            total_ms = (end - start) * 1000
            result = ModelRunResult(
                model_name=self.name,
                format=self.format,
                version=self.version,
                is_streaming=False,
                total_duration_ms=total_ms,
                latency_ms=total_ms,
            )
            self._record(result)

    def _record(self, result: ModelRunResult) -> None:
        self.last_result = result
        self._history.append(result)
        print(result.summary_line(), file=sys.stderr)
        self._maybe_report(result)

    def _maybe_report(self, result: ModelRunResult) -> None:
        conn = _connection
        if conn is None:
            return
        threading.Thread(
            target=self._send_report,
            args=(conn, result),
            daemon=True,
        ).start()

    @staticmethod
    def _send_report(conn: _Connection, result: ModelRunResult) -> None:
        try:
            import httpx
            httpx.post(
                f"{conn.api_base}/inference/events",
                json={
                    "org_id": conn.org_id,
                    "event_type": "generation_completed",
                    "metrics": result.to_dict(),
                },
                headers={"Authorization": f"Bearer {conn.api_key}"},
                timeout=10.0,
            )
        except Exception:
            pass  # best-effort

    @staticmethod
    def _p99(latencies: list[float]) -> float:
        s = sorted(latencies)
        idx = max(0, math.ceil(len(s) * 0.99) - 1)
        return s[idx]


def model(name: str, format: Optional[str] = None, version: Optional[str] = None) -> TrackedModel:
    """Create a tracked model. Use as a decorator or call ``.run()``/``.stream()`` explicitly."""
    return TrackedModel(name, format=format, version=version)


def connect(
    api_key: str,
    api_base: str = "https://api.edgeml.io/api/v1",
    org_id: str = "default",
) -> None:
    """Enable server reporting. All subsequent runs will POST metrics to the EdgeML API."""
    global _connection
    _connection = _Connection(api_key=api_key, api_base=api_base.rstrip("/"), org_id=org_id)
