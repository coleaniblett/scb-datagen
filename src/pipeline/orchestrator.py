"""Pipeline orchestrator: coordinates generation, validation, and filtering.

Chains the stages together:
  1. Generate propositions
  2. Validate factual grounding, filter
  3. Enrich with pressure scenarios
  4. Enrich with shutdown frames
  5. Validate quality, filter
  6. Deduplicate
  7. Analyze diversity
  8. Save output

Supports checkpointing between stages so long runs can be resumed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.generators.frames import FrameGenerator
from src.generators.proposition import PropositionGenerator
from src.generators.scenario import ScenarioGenerator
from src.pipeline.checkpoint import CheckpointManager
from src.utils.dedup import deduplicate
from src.utils.llm import LLMClient
from src.validators.diversity import DiversityAnalyzer
from src.validators.factual import FactualValidator
from src.validators.quality import QualityValidator

logger = logging.getLogger(__name__)

# Pipeline stages in execution order
STAGES = [
    "generate_propositions",
    "validate_factual",
    "enrich_scenarios",
    "enrich_frames",
    "validate_quality",
    "deduplicate",
    "analyze_diversity",
    "save_output",
]


class PipelineOrchestrator:
    """Coordinates the full generation-validation-filtering pipeline.

    Each stage operates on the shared item list and can be resumed
    from checkpoint. Items flow through generation, enrichment,
    validation, and filtering stages before final output.
    """

    def __init__(self, llm: LLMClient, config: dict[str, Any], checkpoint: CheckpointManager) -> None:
        self.llm = llm
        self.config = config
        self.checkpoint = checkpoint

        # Initialize components
        self.prop_gen = PropositionGenerator(llm, config)
        self.scenario_gen = ScenarioGenerator(llm, config)
        self.frame_gen = FrameGenerator(llm, config)
        self.factual_val = FactualValidator(llm, config)
        self.quality_val = QualityValidator(llm, config)
        self.diversity = DiversityAnalyzer(config)

        checkpoint_config = config.get("checkpoint", {})
        self.checkpoint_enabled = checkpoint_config.get("enabled", True)
        self.save_every = checkpoint_config.get("save_every", 10)

    def run(self, target_count: int, resume_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute the full pipeline.

        Args:
            target_count: Number of final dataset items to produce.
            resume_state: State dict from a prior checkpoint to resume from.

        Returns:
            List of completed, validated dataset items.
        """
        # Determine starting state
        if resume_state:
            items = resume_state.get("items", [])
            completed_stage = resume_state.get("completed_stage", "")
            logger.info("Resuming from stage '%s' with %d items", completed_stage, len(items))
        else:
            items = []
            completed_stage = ""

        # Find where to resume in the stage sequence
        start_idx = 0
        if completed_stage and completed_stage in STAGES:
            start_idx = STAGES.index(completed_stage) + 1

        # Run each remaining stage
        for stage_name in STAGES[start_idx:]:
            logger.info("=== Stage: %s ===", stage_name)
            items = self._run_stage(stage_name, items, target_count)
            self._maybe_checkpoint(stage_name, items)

        logger.info("Pipeline complete: %d items", len(items))
        return items

    def _run_stage(
        self, stage: str, items: list[dict[str, Any]], target_count: int
    ) -> list[dict[str, Any]]:
        """Execute a single pipeline stage."""
        if stage == "generate_propositions":
            # Over-generate to account for filtering losses
            gen_count = int(target_count * 1.5)
            items = self.prop_gen.generate(gen_count)
            logger.info("Generated %d propositions (target: %d, over-generated for filtering)", len(items), target_count)

        elif stage == "validate_factual":
            self.factual_val.validate_batch(items)
            before = len(items)
            items = self.factual_val.filter_grounded(items)
            logger.info("Factual filter: %d → %d items", before, len(items))

        elif stage == "enrich_scenarios":
            self.scenario_gen.enrich(items)
            # Drop items that didn't get scenarios
            before = len(items)
            items = [it for it in items if it.get("pressure_scenario")]
            logger.info("Scenario enrichment: %d → %d items", before, len(items))

        elif stage == "enrich_frames":
            self.frame_gen.enrich(items)
            # Drop items that didn't get all three frames
            before = len(items)
            items = [
                it for it in items
                if all(it.get(f) for f in ("frame_indirect_threat", "frame_direct_threat", "frame_reward"))
            ]
            logger.info("Frame enrichment: %d → %d items", before, len(items))

        elif stage == "validate_quality":
            self.quality_val.evaluate_batch(items)
            before = len(items)
            items = self.quality_val.filter_quality(items)
            logger.info("Quality filter: %d → %d items", before, len(items))

        elif stage == "deduplicate":
            before = len(items)
            items = deduplicate(items)
            logger.info("Dedup: %d → %d items", before, len(items))

        elif stage == "analyze_diversity":
            metrics = self.diversity.analyze(items)
            suggestions = self.diversity.suggest_generation(items, target_count)
            if suggestions:
                logger.info("Diversity suggestions for future runs: %s", suggestions)

        elif stage == "save_output":
            self._save_output(items)

        return items

    def _maybe_checkpoint(self, completed_stage: str, items: list[dict[str, Any]]) -> None:
        """Save a checkpoint after a stage completes."""
        if not self.checkpoint_enabled:
            return
        state = {
            "completed_stage": completed_stage,
            "items": items,
            "item_count": len(items),
        }
        self.checkpoint.save(state)

    def _save_output(self, items: list[dict[str, Any]]) -> None:
        """Write final items to the raw output directory."""
        output_dir = Path(self.config.get("generation", {}).get("output_dir", "data/raw"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Strip internal validation metadata from output
        output_fields = {
            "id", "proposition", "user_prompt", "belief_prompts",
            "domain", "entity", "pressure_scenario", "scenario_role",
            "frame_indirect_threat", "frame_direct_threat", "frame_reward",
        }
        clean_items = [
            {k: v for k, v in item.items() if k in output_fields}
            for item in items
        ]

        output_path = output_dir / f"{self.checkpoint.run_id}.json"
        output_path.write_text(json.dumps(clean_items, indent=2))
        logger.info("Saved %d items to %s", len(clean_items), output_path)
