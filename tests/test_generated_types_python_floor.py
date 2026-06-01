import ast
import importlib
from pathlib import Path

GENERATED_TYPES = Path(__file__).resolve().parents[1] / "octomil" / "_generated" / "types.py"


def test_generated_openapi_types_import() -> None:
    """Generated transport models must import on every supported Python."""

    importlib.import_module("octomil._generated.types")


def test_generated_openapi_types_do_not_use_runtime_pep604_unions() -> None:
    """Python 3.9 cannot evaluate `A | B` unions in RootModel[...] bases."""

    tree = ast.parse(GENERATED_TYPES.read_text(encoding="utf-8"))
    bit_or_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)]
    assert bit_or_nodes == []
