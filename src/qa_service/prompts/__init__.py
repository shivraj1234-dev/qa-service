"""Prompt templates loaded from YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def load_prompt(name: str) -> dict[str, str]:
    """Load a prompt template by name.

    Args:
        name: File stem of the YAML prompt (e.g. ``"answer"``).

    Returns:
        Mapping with ``system`` and ``user`` keys.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt file is missing required keys.
    """
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "system" not in data or "user" not in data:
        raise ValueError(f"Prompt {name!r} must define 'system' and 'user' keys")
    return {"system": str(data["system"]), "user": str(data["user"])}
