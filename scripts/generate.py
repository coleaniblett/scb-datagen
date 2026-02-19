#!/usr/bin/env python3
"""CLI entrypoint for the SCB dataset generation pipeline.

Usage:
    python scripts/generate.py [--config CONFIG] [--resume RUN_ID] [--count N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure src/ is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.checkpoint import CheckpointManager
from src.utils.llm import LLMClient, LLMConfig, load_llm_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "src" / "config" / "defaults.yaml"


def load_config(path: Path) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCB dataset generation pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config YAML (default: src/config/defaults.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run ID to resume from checkpoint",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of items to generate (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without generating",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # Initialize LLM client
    llm = load_llm_from_config(config.get("llm", {}))
    logger.info("LLM backend: %s, model: %s", llm.config.backend, llm.config.model)

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.get("checkpoint", {}).get("dir", "data/checkpoints"),
        run_id=args.resume,
    )

    # Resume from checkpoint if requested
    if args.resume:
        state = checkpoint_mgr.load()
        if state:
            logger.info("Resumed run %s with %d items", args.resume, len(state.get("items", [])))
        else:
            logger.warning("No checkpoint found for run %s, starting fresh", args.resume)

    target_count = args.count or config.get("dataset", {}).get("target_count", 500)
    logger.info("Target: %d items", target_count)

    if args.dry_run:
        logger.info("Dry run complete — config is valid.")
        return

    # TODO: Wire up generators and orchestrator here
    logger.info("Pipeline skeleton ready. Generators not yet implemented.")


if __name__ == "__main__":
    main()
