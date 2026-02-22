"""Unified model resolution with ``model:variant`` syntax.

Parse Ollama-style model specifiers, resolve to engine-specific artifacts,
and provide a single catalog replacing per-engine hardcoded sets.

Usage::

    from edgeml.models import parse, resolve, list_models

    parsed = parse("gemma-3b:4bit")
    resolved = resolve("gemma-3b:4bit", available_engines=["mlx-lm", "llama.cpp"])
"""

from .catalog import CATALOG, list_models
from .parser import ParsedModel, parse
from .resolver import ResolvedModel, resolve

__all__ = [
    "CATALOG",
    "ParsedModel",
    "ResolvedModel",
    "list_models",
    "parse",
    "resolve",
]
