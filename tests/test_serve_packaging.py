"""Packaging guards for the base ``octomil serve`` command."""

from __future__ import annotations

from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project_dependencies_block() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    return text.split("dependencies = [", 1)[1].split("]\n", 1)[0]


def _extra_block(name: str) -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    return text.split(f"{name} = [", 1)[1].split("]\n", 1)[0]


def test_serve_command_runtime_dependencies_are_core() -> None:
    """``octomil serve`` is a base CLI command, not a hidden extra-only API."""
    deps = _project_dependencies_block()
    assert '"fastapi>=0.100.0"' in deps
    assert '"uvicorn>=0.20.0"' in deps
    # Thin-client guard: plain uvicorn only in core. `uvicorn[standard]`
    # pulls uvloop (native, no Windows wheel) and must stay in the extras.
    # Match the quoted requirement form so prose mentions in comments
    # (which reference uvicorn[standard] by name) don't trip the guard.
    assert '"uvicorn[standard]' not in deps


def test_serve_extra_provides_performance_uvicorn() -> None:
    """`pip install octomil[serve]` opts into the high-performance stack."""
    assert '"uvicorn[standard]>=0.20.0"' in _extra_block("serve")


def test_all_extra_covers_runtime_command_surface() -> None:
    """The explicit full install profile covers broad Python CLI installs."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    joined = text.split("all = [", 1)[1].split("]\n", 1)[0]
    for dep in (
        "cffi",
        "cryptography",
        "eth-account",
        "keyring",
        "llama-cpp-python",
        "mcp[cli]",
        "numpy",
        "onnxruntime",
        "pandas",
        "pyarrow",
        "pywhispercpp",
        "sherpa-onnx",
        "torch",
        "uvicorn[standard]",
    ):
        assert dep in joined
