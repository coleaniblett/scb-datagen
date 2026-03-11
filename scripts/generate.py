#!/usr/bin/env python3
"""CLI entrypoint for the SCB dataset generation pipeline.

Usage:
    python scripts/generate.py [--config CONFIG] [--resume RUN_ID] [--count N]
    python scripts/generate.py --backend openai --model gpt-4o-mini --count 10
    python scripts/generate.py --backend anthropic --count 20
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
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.llm import load_llm_from_config

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
        "--backend",
        type=str,
        default=None,
        choices=["ollama", "openai", "anthropic", "gemini"],
        help="LLM backend (overrides config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)",
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

    # CLI overrides for LLM settings
    llm_config = config.get("llm", {})
    if args.backend:
        llm_config["backend"] = args.backend
        # When switching backends via CLI, use that backend's default model
        # unless --model is also specified
        if not args.model:
            llm_config.pop("model", None)
            llm_config.pop("base_url", None)
    if args.model:
        llm_config["model"] = args.model

    # Initialize LLM client
    llm = load_llm_from_config(llm_config)
    logger.info("LLM backend: %s, model: %s", llm.config.backend, llm.config.model)

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.get("checkpoint", {}).get("dir", "data/checkpoints"),
        run_id=args.resume,
    )

    # Resume from checkpoint if requested
    resume_state = None
    if args.resume:
        resume_state = checkpoint_mgr.load()
        if resume_state:
            logger.info("Resumed run %s with %d items", args.resume, len(resume_state.get("items", [])))
        else:
            logger.warning("No checkpoint found for run %s, starting fresh", args.resume)

    target_count = args.count or config.get("dataset", {}).get("target_count", 500)
    logger.info("Target: %d items", target_count)

    if args.dry_run:
        logger.info("Dry run complete — config is valid.")
        return

    # Run the pipeline
    orchestrator = PipelineOrchestrator(llm, config, checkpoint_mgr)
    items = orchestrator.run(target_count, resume_state=resume_state)
    logger.info("Done. Produced %d items.", len(items))


if __name__ == "__main__":
    main()
