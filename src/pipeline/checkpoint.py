"""Checkpoint support for resuming interrupted pipeline runs.

Saves intermediate state (generated items, validation results) to disk
so that long-running generation can be resumed without losing progress.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = "data/checkpoints"


class CheckpointManager:
    """Manages saving and loading pipeline checkpoints."""

    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR, run_id: str | None = None) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
            run_id: Identifier for this pipeline run. If None, a
                    timestamp-based ID is generated.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.run_id}.json"

    def save(self, state: dict[str, Any]) -> Path:
        """Save pipeline state to a checkpoint file.

        Args:
            state: Arbitrary dict representing current pipeline state.
                   Should include items generated so far, current step, etc.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
        }
        path = self._checkpoint_path
        path.write_text(json.dumps(checkpoint, indent=2, default=str))
        logger.info("Checkpoint saved: %s (%d items)", path, len(state.get("items", [])))
        return path

    def load(self) -> dict[str, Any] | None:
        """Load the most recent checkpoint for this run.

        Returns:
            The saved state dict, or None if no checkpoint exists.
        """
        path = self._checkpoint_path
        if not path.exists():
            logger.info("No checkpoint found for run %s", self.run_id)
            return None

        data = json.loads(path.read_text())
        logger.info("Loaded checkpoint: %s", path)
        return data.get("state")

    def list_runs(self) -> list[str]:
        """List all available checkpoint run IDs.

        Returns:
            Sorted list of run IDs with saved checkpoints.
        """
        return sorted(p.stem for p in self.checkpoint_dir.glob("*.json"))
