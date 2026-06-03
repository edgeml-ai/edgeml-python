"""Packaging guards for the base ``octomil serve`` command."""

from __future__ import annotations

from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project_dependencies_block() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    return text.split("dependencies = [", 1)[1].split("]\n", 1)[0]


def test_serve_command_runtime_dependencies_are_core() -> None:
    """``octomil serve`` is a base CLI command, not a hidden extra-only API."""
    deps = _project_dependencies_block()
    assert '"fastapi>=0.100.0"' in deps
    assert '"uvicorn[standard]>=0.20.0"' in deps


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
    ):
        assert dep in joined
