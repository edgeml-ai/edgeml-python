"""``octomil doctor`` — diagnose planner / cache / runtime / auth state.

Embedded callers (Ren'Py games, kiosk apps, PyInstaller binaries)
hit a long tail of confusing failure modes: ``sqlite3`` missing,
``OCTOMIL_SERVER_KEY`` unset, ``OCTOMIL_API_BASE`` accidentally
pointing at staging, native runtime dylib missing, planner returning
401 because the org id is mismatched. Each one surfaces today as a
generic ``RUNTIME_UNAVAILABLE`` deep inside an inference call.

``octomil doctor`` walks every diagnostic the SDK can run locally
and prints one structured report:

  - Auth: which env vars are set, which are missing, how the
    planner resolves them. Never prints the key itself.
  - Planner: cache backend in use (sqlite vs memory vs null),
    cache directory, network reachability of the configured
    ``OCTOMIL_API_BASE``.
  - Local runtimes: which engine extras / native capabilities are available
    (native TTS, mlx-lm, llama.cpp, whisper.cpp, onnxruntime), which models are
    staged, which static recipes are available offline.
  - Cache directories: artifact cache root, free space, sample of
    materialized artifacts.

Each row prints ``OK`` / ``WARN`` / ``ERROR`` so a developer scanning
the output knows where to look. Exit code is 0 when everything is
``OK`` or ``WARN``; non-zero when any ``ERROR`` row appears.
"""

from __future__ import annotations

import os
import shutil
import sys

import click

_OK = "OK"
_WARN = "WARN"
_ERROR = "ERROR"


def _row(status: str, label: str, detail: str = "") -> tuple[str, str, str]:
    return (status, label, detail)


def _check_python_runtime() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    rows.append(_row(_OK, "python", f"{sys.version.split()[0]} ({sys.executable})"))
    rows.append(_row(_OK, "platform", f"{sys.platform}"))
    try:
        import sqlite3 as _sqlite3  # noqa: F401

        rows.append(_row(_OK, "sqlite3", "available (planner can use on-disk cache)"))
    except ImportError:
        rows.append(
            _row(
                _WARN,
                "sqlite3",
                "missing (planner falls back to in-memory cache; this is "
                "expected on Ren'Py / sandboxed CPython / some PyInstaller builds)",
            )
        )
    return rows


def _check_auth() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    server_key = os.environ.get("OCTOMIL_SERVER_KEY")
    api_key = os.environ.get("OCTOMIL_API_KEY")
    org_id = os.environ.get("OCTOMIL_ORG_ID")
    api_base = os.environ.get("OCTOMIL_API_BASE")
    publishable = os.environ.get("OCTOMIL_PUBLISHABLE_KEY")

    if server_key:
        rows.append(_row(_OK, "OCTOMIL_SERVER_KEY", "set (planner uses Bearer auth)"))
    elif api_key:
        rows.append(_row(_OK, "OCTOMIL_API_KEY", "set (legacy; OCTOMIL_SERVER_KEY preferred)"))
    elif publishable:
        rows.append(
            _row(
                _WARN,
                "auth",
                "OCTOMIL_PUBLISHABLE_KEY only — public-client mode; planner "
                "calls that need server auth will be skipped",
            )
        )
    else:
        rows.append(
            _row(
                _WARN,
                "auth",
                "no OCTOMIL_SERVER_KEY / OCTOMIL_API_KEY set; planner will "
                "fall back to local routing only. Set OCTOMIL_SERVER_KEY for "
                "server-resolved app refs.",
            )
        )

    if org_id:
        rows.append(_row(_OK, "OCTOMIL_ORG_ID", "set"))
    elif server_key or api_key:
        rows.append(
            _row(
                _WARN,
                "OCTOMIL_ORG_ID",
                "not set; planner Bearer auth resolves org server-side, but "
                "explicit OCTOMIL_ORG_ID disambiguates multi-org users",
            )
        )

    if api_base:
        rows.append(_row(_OK, "OCTOMIL_API_BASE", api_base))
    else:
        rows.append(_row(_OK, "OCTOMIL_API_BASE", "default (https://api.octomil.com)"))

    return rows


def _check_planner_cache() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    cache_env = os.environ.get("OCTOMIL_RUNTIME_PLANNER_CACHE", "").strip()
    if cache_env == "0":
        rows.append(_row(_WARN, "planner cache", "disabled by OCTOMIL_RUNTIME_PLANNER_CACHE=0"))
        return rows

    try:
        from octomil.runtime.planner.store import (
            MemoryRuntimePlannerStore,
            SQLiteRuntimePlannerStore,
            build_runtime_planner_store,
        )

        store = build_runtime_planner_store()
        if isinstance(store, SQLiteRuntimePlannerStore):
            db = os.environ.get("OCTOMIL_RUNTIME_PLANNER_DB") or "~/.cache/octomil/runtime_planner.sqlite3"
            rows.append(_row(_OK, "planner cache", f"sqlite ({db})"))
        elif isinstance(store, MemoryRuntimePlannerStore):
            rows.append(
                _row(
                    _WARN,
                    "planner cache",
                    "in-memory (sqlite3 unavailable; cache is per-process)",
                )
            )
        else:
            rows.append(_row(_OK, "planner cache", type(store).__name__))
        store.close()
    except ImportError as exc:
        rows.append(_row(_WARN, "planner cache", f"unavailable: {exc}"))
    return rows


def _check_artifact_cache() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    cache_dir = os.environ.get("OCTOMIL_CACHE_DIR")
    if not cache_dir:
        from pathlib import Path

        cache_dir = str(Path.home() / ".cache" / "octomil")
    rows.append(_row(_OK, "artifact cache", cache_dir))
    try:
        usage = shutil.disk_usage(cache_dir if os.path.exists(cache_dir) else os.path.dirname(cache_dir))
        free_gb = usage.free / (1024**3)
        status = _OK if free_gb >= 2.0 else _WARN
        rows.append(_row(status, "free space", f"{free_gb:.1f} GiB available"))
    except OSError:
        # Either the parent dir doesn't exist or the platform can't
        # report disk usage. Not a blocker for diagnostic output.
        rows.append(_row(_WARN, "free space", "unable to determine"))
    return rows


def _check_local_engines() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    try:
        from octomil.runtime.native.capabilities import CAPABILITY_CHAT_COMPLETION, CAPABILITY_CHAT_STREAM
        from octomil.runtime.native.loader import NativeRuntime
        from octomil.runtime.native.tts_batch_backend import runtime_advertises_tts_batch
        from octomil.runtime.native.tts_stream_backend import runtime_advertises_tts_stream

        runtime = NativeRuntime.open()
        try:
            caps = runtime.capabilities()
            batch = runtime_advertises_tts_batch(runtime)
            stream = runtime_advertises_tts_stream(runtime)
        finally:
            runtime.close()
        rows.append(
            _row(
                _OK,
                "native runtime dylib",
                os.environ.get("OCTOMIL_RUNTIME_DYLIB", "auto-discovered"),
            )
        )
        rows.append(
            _row(
                _OK if caps.supported_engines else _WARN,
                "native engines",
                ", ".join(caps.supported_engines) if caps.supported_engines else "none advertised",
            )
        )
        rows.append(
            _row(
                _OK if caps.supported_capabilities else _WARN,
                "native capabilities",
                ", ".join(caps.supported_capabilities) if caps.supported_capabilities else "none advertised",
            )
        )
        chat_ready = caps.claims(CAPABILITY_CHAT_COMPLETION)
        chat_detail = "chat.completion"
        if caps.claims(CAPABILITY_CHAT_STREAM):
            chat_detail += ", chat.stream"
        if not any(engine in caps.supported_engines for engine in ("llama_cpp", "llama.cpp")):
            chat_detail += "; llama.cpp engine not advertised"
            chat_ready = False
        rows.append(
            _row(
                _OK if chat_ready else _WARN,
                "native chat runtime",
                chat_detail if chat_ready else f"not ready ({chat_detail})",
            )
        )
        status = _OK if batch or stream else _WARN
        tts_caps_detail = ", ".join(
            name for name, available in (("audio.tts.batch", batch), ("audio.tts.stream", stream)) if available
        )
        rows.append(_row(status, "native TTS runtime", tts_caps_detail or "not advertised (local TTS unavailable)"))
    except Exception as exc:
        rows.append(_row(_WARN, "native TTS runtime", f"unavailable: {exc}"))

    # ``label`` is the user-facing name (matches the pip extra /
    # PyPI distribution); ``module`` is the actual Python import
    # name.
    engines = [
        ("mlx_lm", "mlx_lm", "chat / responses on Apple Silicon"),
        ("llama_cpp", "llama_cpp", "chat / responses (GGUF)"),
        ("pywhispercpp", "pywhispercpp", "transcription"),
        ("onnxruntime", "onnxruntime", "ONNX classifiers"),
    ]
    for label, module, what in engines:
        try:
            __import__(module)
            rows.append(_row(_OK, label, what))
        except ImportError:
            rows.append(_row(_WARN, label, f"not installed ({what} → cloud only)"))
    return rows


def _check_static_recipes() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    try:
        from octomil.runtime.lifecycle.static_recipes import _RECIPES

        if not _RECIPES:
            rows.append(_row(_WARN, "static recipes", "none registered"))
        else:
            rows.append(
                _row(
                    _OK,
                    "static recipes",
                    f"{len(_RECIPES)} canonical models available offline "
                    f"({', '.join(sorted({m for m, _ in _RECIPES}))})",
                )
            )
    except ImportError as exc:
        rows.append(_row(_WARN, "static recipes", f"unavailable: {exc}"))
    return rows


def _print_section(title: str, rows: list[tuple[str, str, str]]) -> bool:
    """Return True iff any row in this section is ERROR."""
    click.echo(f"\n{title}")
    click.echo("-" * len(title))
    has_error = False
    for status, label, detail in rows:
        if status == _ERROR:
            has_error = True
            color = "red"
        elif status == _WARN:
            color = "yellow"
        else:
            color = "green"
        prefix = click.style(f"  [{status:5}]", fg=color)
        line = f"{prefix} {label:24}  {detail}"
        click.echo(line)
    return has_error


@click.command("doctor")
def doctor_cmd() -> None:
    """Diagnose planner / cache / runtime / auth state.

    Prints a structured report covering Python runtime, auth env
    vars, planner cache backend, artifact cache, installed local
    engines, and registered static recipes. Exit code is non-zero
    only when any check reports ERROR; WARN keeps exit 0 so the
    command works in CI gates that only fail on real breakage.
    """
    from octomil.venv_reexec import try_managed_venv_reexec

    try_managed_venv_reexec(include_non_frozen=True)

    click.echo("octomil doctor: scanning local SDK state\n")

    any_error = False
    any_error |= _print_section("Python runtime", _check_python_runtime())
    any_error |= _print_section("Auth", _check_auth())
    any_error |= _print_section("Planner cache", _check_planner_cache())
    any_error |= _print_section("Artifact cache", _check_artifact_cache())
    any_error |= _print_section("Local engines", _check_local_engines())
    any_error |= _print_section("Static offline recipes", _check_static_recipes())

    click.echo("")
    if any_error:
        click.echo(click.style("octomil doctor: errors detected (see ERROR rows above)", fg="red"))
        sys.exit(1)
    click.echo(click.style("octomil doctor: scan complete", fg="green"))


def register(cli: click.Group) -> None:
    cli.add_command(doctor_cmd)
