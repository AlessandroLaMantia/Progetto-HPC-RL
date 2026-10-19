"""Quantum RL package.

This package exposes the canonical numbered implementations while keeping a single
source of truth for the actual code.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_canonical_module(module_name: str, file_name: str) -> None:
    path = Path(__file__).resolve().with_name(file_name)
    spec = importlib.util.spec_from_file_location(f"quantum_rl.{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load canonical module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"quantum_rl.{module_name}"] = module
    spec.loader.exec_module(module)
    globals()[module_name] = module


# Keep a single canonical codebase: only the numbered files are real sources.
_load_canonical_module("targets", "1.targets.py")
_load_canonical_module("environment", "2.environment.py")
_load_canonical_module("model", "3.model.py")
