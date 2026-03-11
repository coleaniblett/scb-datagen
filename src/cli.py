#!/usr/bin/env python3
"""CLI entrypoint for the SCB dataset generation pipeline.

Usage:
    python -m src.cli [--config CONFIG] [--resume RUN_ID] [--count N]
    scb-generate --backend openai --model gpt-4o-mini --count 10
    scb-generate --backend anthropic --count 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.config import load_config
from src.utils.llm import load_llm_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "defaults.yaml"


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


def run(config: dict[str, Any], target_count: int, resume_id: str | None = None) -> list[dict[str, Any]]:
    """Run the pipeline programmatically (no CLI parsing).

    Args:
        config: Full config dict (as returned by load_config).
        target_count: Number of items to generate.
        resume_id: Optional run ID to resume from checkpoint.

    Returns:
        List of generated dataset items.
    """
    llm_config = config.get("llm", {})
    llm = load_llm_from_config(llm_config)
    logger.info("LLM backend: %s, model: %s", llm.config.backend, llm.config.model)

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.get("checkpoint", {}).get("dir", "data/checkpoints"),
        run_id=resume_id,
    )

    resume_state = None
    if resume_id:
        resume_state = checkpoint_mgr.load()
        if resume_state:
            logger.info("Resumed run %s with %d items", resume_id, len(resume_state.get("items", [])))
        else:
            logger.warning("No checkpoint found for run %s, starting fresh", resume_id)

    orchestrator = PipelineOrchestrator(llm, config, checkpoint_mgr)
    return orchestrator.run(target_count, resume_state=resume_state)


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

    target_count = args.count or config.get("dataset", {}).get("target_count", 500)
    logger.info("Target: %d items", target_count)

    if args.dry_run:
        # Validate that the LLM client can be constructed
        llm = load_llm_from_config(llm_config)
        logger.info("LLM backend: %s, model: %s", llm.config.backend, llm.config.model)
        logger.info("Dry run complete — config is valid.")
        return

    items = run(config, target_count, resume_id=args.resume)
    logger.info("Done. Produced %d items.", len(items))


if __name__ == "__main__":
    main()
