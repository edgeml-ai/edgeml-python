"""Generated-output cache for regenerable SDK results.

This cache is intentionally separate from the provisioned artifact cache:
model weights live in durable storage and should not be evicted just because
voice clips or embeddings were generated. Generated outputs are regenerable,
so they live in the OS cache directory and are LRU-evicted by byte budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_MAX_BYTES_ENV = "OCTOMIL_OUTPUT_CACHE_MAX_BYTES"
_CACHE_DIR_ENV = "OCTOMIL_OUTPUT_CACHE_DIR"


def user_cache_dir() -> Path:
    """Return the OS-purgeable cache root for Octomil generated outputs."""
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "octomil"
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
        return Path(root or Path.home() / "AppData" / "Local") / "Octomil" / "Cache"
    root = os.environ.get("XDG_CACHE_HOME")
    return Path(root).expanduser() / "octomil" if root else Path.home() / ".cache" / "octomil"


def configured_max_bytes() -> int:
    raw = os.environ.get(_MAX_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BYTES
    return max(0, value)


def derive_output_key(
    capability: str,
    *,
    model: str,
    payload: Mapping[str, Any],
    schema_version: str = "v1",
) -> str:
    """Return a content-addressed key without exposing raw payload in paths."""
    h = hashlib.sha256()
    h.update(schema_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(capability.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h.update(encoded.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class CachedOutput:
    data: bytes
    metadata: dict[str, Any]


class GeneratedOutputCache:
    """Byte-budgeted file cache for generated outputs."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.root = root or user_cache_dir() / "outputs"
        self.max_bytes = configured_max_bytes() if max_bytes is None else max(0, max_bytes)

    def get(self, capability: str, key: str) -> CachedOutput | None:
        data_path, meta_path = self._paths(capability, key)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            data = data_path.read_bytes()
        except Exception:
            self._remove_pair(data_path, meta_path)
            return None

        now = time.time()
        try:
            os.utime(data_path, (now, now))
            os.utime(meta_path, (now, now))
        except OSError:
            pass
        return CachedOutput(data=data, metadata=metadata if isinstance(metadata, dict) else {})

    def put(
        self,
        capability: str,
        key: str,
        data: bytes,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.max_bytes <= 0 or len(data) > self.max_bytes:
            return

        data_path, meta_path = self._paths(capability, key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(data_path, data)
        self._atomic_write(
            meta_path,
            json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        self._evict_to_budget()

    def clear(self, capability: str | None = None) -> None:
        root = self.root / capability if capability else self.root
        if not root.exists():
            return
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    def _paths(self, capability: str, key: str) -> tuple[Path, Path]:
        safe_capability = capability.replace("/", "_").replace("..", "_")
        safe_key = "".join(ch for ch in key.lower() if ch in "0123456789abcdef")
        if len(safe_key) != 64:
            raise ValueError("generated-output cache key must be a 64-character hex digest")
        base = self.root / safe_capability / safe_key[:2] / safe_key
        return base.with_suffix(".bin"), base.with_suffix(".json")

    def _evict_to_budget(self) -> None:
        if not self.root.exists():
            return
        entries: list[tuple[float, Path, Path, int]] = []
        total = 0
        for data_path in self.root.rglob("*.bin"):
            meta_path = data_path.with_suffix(".json")
            try:
                size = data_path.stat().st_size + (meta_path.stat().st_size if meta_path.exists() else 0)
                mtime = data_path.stat().st_mtime
            except OSError:
                continue
            total += size
            entries.append((mtime, data_path, meta_path, size))

        if total <= self.max_bytes:
            return

        for _, data_path, meta_path, size in sorted(entries, key=lambda item: item[0]):
            self._remove_pair(data_path, meta_path)
            total -= size
            if total <= self.max_bytes:
                break

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        with NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    @staticmethod
    def _remove_pair(data_path: Path, meta_path: Path) -> None:
        for path in (data_path, meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
