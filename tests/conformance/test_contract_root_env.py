from pathlib import Path

from tests.conformance.conftest import _contract_root_from_env


def test_contract_root_env_supplies_conformance_cache(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "contract"
    (root / "conformance").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts" / "generate_conformance.py").write_text("# test generator\n")

    monkeypatch.setenv("CONTRACT_ROOT", str(root))

    assert _contract_root_from_env() == root


def test_contract_root_env_rejects_incomplete_layout(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "contract"
    root.mkdir()

    monkeypatch.setenv("CONTRACT_ROOT", str(root))

    assert _contract_root_from_env() is None
