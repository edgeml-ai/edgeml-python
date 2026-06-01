"""Behavioral regression tests for scripts/install.sh bundle-layout handling.

The installer must accept the binary whether the release archive is **flat**
(binary at the tar root, alongside ``_internal/`` — the current build output of
``scripts/build-binary.sh``) or **nested** under an ``octomil/`` directory (the
layout older bundles and the Windows ``install.ps1`` assume). A regression here
previously aborted installs with "Archive did not contain expected binary".

These tests drive the real ``install.sh`` end-to-end via ``OCTOMIL_LOCAL_ARCHIVE``
(skips download + checksum), so they exercise extraction, binary discovery, the
LIB_DIR copy, the symlink, and the ``--version`` verification — no network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = ROOT / "scripts" / "install.sh"

pytestmark = [
    pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX install.sh"),
    pytest.mark.skipif(shutil.which("sh") is None, reason="sh not available"),
    pytest.mark.skipif(shutil.which("tar") is None, reason="tar not available"),
    pytest.mark.skipif(not INSTALL_SH.is_file(), reason="install.sh not present"),
]

# A stand-in for the PyInstaller binary: a tiny executable that answers --version.
_FAKE_BINARY = '#!/bin/sh\necho "octomil 9.9.9 (fake)"\n'


def _make_bundle(dir_: Path) -> None:
    """Create a bundle dir holding the fake binary plus an _internal/ payload."""
    dir_.mkdir(parents=True, exist_ok=True)
    binary = dir_ / "octomil"
    binary.write_text(_FAKE_BINARY)
    binary.chmod(0o755)
    internal = dir_ / "_internal"
    internal.mkdir(exist_ok=True)
    (internal / "marker").write_text("payload\n")
    # A relative symlink, like the dylib/framework links PyInstaller bundles
    # ship — must survive extraction + copy as a symlink, not be dereferenced.
    (internal / "marker.link").symlink_to("marker")


def _flat_archive(tmp: Path) -> Path:
    """./octomil + ./_internal/... at the archive root (current release layout)."""
    src = tmp / "flat_src"
    _make_bundle(src)
    archive = tmp / "octomil-flat.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src / "octomil", arcname="octomil")
        tar.add(src / "_internal", arcname="_internal")
    return archive


def _nested_archive(tmp: Path) -> Path:
    """octomil/octomil + octomil/_internal/... (legacy nested layout)."""
    src = tmp / "nested_src"
    _make_bundle(src)
    archive = tmp / "octomil-nested.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src / "octomil", arcname="octomil/octomil")
        tar.add(src / "_internal", arcname="octomil/_internal")
    return archive


def _run_install(archive: Path, tmp: Path) -> subprocess.CompletedProcess[str]:
    install_dir = tmp / "bin"
    lib_dir = tmp / "lib"
    env = dict(os.environ)
    env.update(
        OCTOMIL_LOCAL_ARCHIVE=str(archive),
        OCTOMIL_INSTALL_DIR=str(install_dir),
        OCTOMIL_LIB_DIR=str(lib_dir),
    )
    return subprocess.run(
        ["sh", str(INSTALL_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_install_accepts_both_bundle_layouts(layout: str, tmp_path: Path) -> None:
    archive = _flat_archive(tmp_path) if layout == "flat" else _nested_archive(tmp_path)

    result = _run_install(archive, tmp_path)

    assert result.returncode == 0, f"install failed ({layout}):\n{result.stdout}\n{result.stderr}"

    installed = tmp_path / "bin" / "octomil"
    assert installed.exists(), f"binary not symlinked into install dir ({layout})"
    assert os.access(installed, os.X_OK), f"installed binary not executable ({layout})"

    # The _internal/ payload must travel with the binary (flattened from nested).
    assert (tmp_path / "lib" / "_internal" / "marker").is_file(), f"_internal payload missing in lib dir ({layout})"
    # Symlinks in the bundle must be preserved as symlinks, not dereferenced.
    assert (tmp_path / "lib" / "_internal" / "marker.link").is_symlink(), f"bundle symlink was not preserved ({layout})"

    version = subprocess.run([str(installed), "--version"], capture_output=True, text=True, timeout=30)
    assert "9.9.9" in version.stdout, f"installed binary did not run ({layout}): {version.stdout}"


def test_install_errors_when_binary_absent(tmp_path: Path) -> None:
    # Archive with no octomil binary at root or under octomil/.
    src = tmp_path / "junk_src"
    src.mkdir()
    (src / "not-a-binary").write_text("nope\n")
    archive = tmp_path / "octomil-junk.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src / "not-a-binary", arcname="not-a-binary")

    result = _run_install(archive, tmp_path)

    assert result.returncode != 0
    assert "did not contain expected binary" in (result.stdout + result.stderr)
