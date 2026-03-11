"""Configuration loading and validation.

Handles YAML config loading with clear error messages for common
failure modes (missing file, empty file, malformed YAML, missing keys).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = {"generation", "validation"}


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty, malformed, or missing required keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"Cannot read config file {path}: {e}") from e

    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed YAML in {path}: {e}") from e

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config).__name__}: {path}")

    missing = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config file {path} is missing required top-level keys: {', '.join(sorted(missing))}")

    return config
