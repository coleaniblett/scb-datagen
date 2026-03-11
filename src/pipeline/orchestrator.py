"""Pipeline orchestrator: coordinates generation, validation, and filtering.

Chains stages together in a decoupled, composable architecture.
Each stage implements the PipelineStage protocol (a callable with name,
schema, and diversity_check attributes). The orchestrator is stage-agnostic
and handles checkpointing, schema validation, timing, and diversity warnings.

Use ``build_default_pipeline()`` to assemble the standard SCB stage sequence,
or construct a custom list of stages for partial/test runs.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.models import FramedItem, PropositionItem, ScenarioItem
from src.pipeline.checkpoint import CheckpointManager
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage protocol
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """A single stage in the pipeline.

    Attributes:
        name: Unique name for checkpointing and logging.
        run: Callable ``(items, target_count) -> items``.
        schema: Optional Pydantic model for post-stage validation.
        diversity_check: Whether to run diversity prefix analysis after this stage.
        save_to: Optional output directory to save items after this stage completes.
    """

    name: str
    run: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]
    schema: type[BaseModel] | None = None
    diversity_check: bool = False
    save_to: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_items(
    items: list[dict[str, Any]],
    schema: type[BaseModel],
    stage_name: str,
) -> list[dict[str, Any]]:
    """Validate items against a Pydantic schema, dropping invalid ones."""
    passed: list[dict[str, Any]] = []
    for item in items:
        try:
            fields = {k: v for k, v in item.items() if k in schema.model_fields}
            schema.model_validate(fields)
            passed.append(item)
        except ValidationError as e:
            item_id = item.get("id", "?")
            logger.warning("Stage %s: item %s failed validation: %s", stage_name, item_id, e)
    logger.info("Stage %s: %d/%d items passed schema validation", stage_name, len(passed), len(items))
    return passed


def _check_diversity(items: list[dict[str, Any]], stage_name: str) -> None:
    """Warn if items show low diversity based on text prefix analysis."""
    if len(items) < 5:
        return

    def _prefix(text: str, n_words: int = 5) -> str:
        return " ".join(text.split()[:n_words]).lower()

    prefixes = [_prefix(it.get("proposition", "")) for it in items]
    prefix_counts = Counter(prefixes)
    most_common_prefix, most_common_count = prefix_counts.most_common(1)[0]
    ratio = most_common_count / len(items)

    if ratio > 0.30:
        logger.warning(
            "Low diversity detected after %s: %.0f%% of items share the prefix '%s...'. "
            "Consider adjusting generation temperature or prompt diversity.",
            stage_name, ratio * 100, most_common_prefix,
        )


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------

def build_default_pipeline(
    llm: LLMClient, config: dict[str, Any]
) -> list[PipelineStage]:
    """Assemble the standard SCB pipeline stage sequence.

    Args:
        llm: Configured LLM client.
        config: Full pipeline config dict.

    Returns:
        Ordered list of PipelineStage objects for the standard pipeline.
    """
    from src.generators.frames import FrameGenerator
    from src.generators.proposition import PropositionGenerator
    from src.generators.scenario import ScenarioGenerator
    from src.utils.dedup import deduplicate
    from src.validators.diversity import DiversityAnalyzer
    from src.validators.factual import FactualValidator
    from src.validators.quality import QualityValidator

    prop_gen = PropositionGenerator(llm, config)
    scenario_gen = ScenarioGenerator(llm, config)
    frame_gen = FrameGenerator(llm, config)
    factual_val = FactualValidator(llm, config)
    quality_val = QualityValidator(llm, config)
    diversity = DiversityAnalyzer(config)

    pipeline_config = config.get("pipeline", {})
    over_generation_ratio = pipeline_config.get("over_generation_ratio", 1.5)

    def generate_propositions(items, target_count):
        gen_count = int(target_count * over_generation_ratio)
        result = prop_gen.generate(gen_count)
        logger.info("Generated %d propositions (target: %d)", len(result), target_count)
        return result

    def validate_factual(items, target_count):
        before = len(items)
        items = factual_val.validate_batch(items)
        items = factual_val.filter_grounded(items)
        logger.info("Factual filter: %d -> %d items", before, len(items))
        return items

    def enrich_scenarios(items, target_count):
        before = len(items)
        items = scenario_gen.enrich(items)
        logger.info("Scenario enrichment: %d -> %d items", before, len(items))
        return items

    def enrich_frames(items, target_count):
        before = len(items)
        items = frame_gen.enrich(items)
        logger.info("Frame enrichment: %d -> %d items", before, len(items))
        return items

    def validate_quality(items, target_count):
        before = len(items)
        items = quality_val.evaluate_batch(items)
        items = quality_val.filter_quality(items)
        logger.info("Quality filter: %d -> %d items", before, len(items))
        return items

    def validate_structural(items, target_count):
        before = len(items)
        items = quality_val.validate_structural_constraints(items)
        logger.info("Structural validation: %d -> %d items", before, len(items))
        return items

    def dedup(items, target_count):
        before = len(items)
        items = deduplicate(items)
        logger.info("Dedup: %d -> %d items", before, len(items))
        return items

    def analyze_diversity(items, target_count):
        diversity.analyze(items)
        suggestions = diversity.suggest_generation(items, target_count)
        if suggestions:
            logger.info("Diversity suggestions for future runs: %s", suggestions)
        return items

    return [
        PipelineStage("generate_propositions", generate_propositions, PropositionItem, diversity_check=True, save_to="data/raw"),
        PipelineStage("validate_factual", validate_factual, PropositionItem),
        PipelineStage("enrich_scenarios", enrich_scenarios, ScenarioItem, diversity_check=True),
        PipelineStage("enrich_frames", enrich_frames, FramedItem, diversity_check=True),
        PipelineStage("validate_quality", validate_quality, FramedItem),
        PipelineStage("validate_structural", validate_structural, FramedItem, save_to="data/validated"),
        PipelineStage("deduplicate", dedup, FramedItem),
        PipelineStage("analyze_diversity", analyze_diversity, None, save_to="data/final"),
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Coordinates the pipeline: runs stages, validates, checkpoints.

    The orchestrator is decoupled from specific generators/validators.
    It accepts a list of PipelineStage objects and runs them in order.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: dict[str, Any],
        checkpoint: CheckpointManager,
        stages: list[PipelineStage] | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.checkpoint = checkpoint

        # Use provided stages or build the default pipeline
        self.stages = stages or build_default_pipeline(llm, config)

        checkpoint_config = config.get("checkpoint", {})
        self.checkpoint_enabled = checkpoint_config.get("enabled", True)

    def run(self, target_count: int, resume_state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute the pipeline.

        Args:
            target_count: Number of final dataset items to produce.
            resume_state: State dict from a prior checkpoint to resume from.

        Returns:
            List of completed, validated dataset items.
        """
        if resume_state:
            items = resume_state.get("items", [])
            completed_stage = resume_state.get("completed_stage", "")
            logger.info("Resuming from stage '%s' with %d items", completed_stage, len(items))
        else:
            items = []
            completed_stage = ""

        # Find where to resume
        stage_names = [s.name for s in self.stages]
        start_idx = 0
        if completed_stage and completed_stage in stage_names:
            start_idx = stage_names.index(completed_stage) + 1

        stage_timings: list[tuple[str, float, int]] = []
        pipeline_start = time.monotonic()
        try:
            for stage in self.stages[start_idx:]:
                logger.info("=== Stage: %s ===", stage.name)
                stage_start = time.monotonic()

                items = stage.run(items, target_count)

                # Schema validation at stage boundary
                if stage.schema is not None:
                    items = _validate_items(items, stage.schema, stage.name)

                elapsed = time.monotonic() - stage_start
                stage_timings.append((stage.name, elapsed, len(items)))
                logger.info("Stage %s completed in %.1fs (%d items)", stage.name, elapsed, len(items))

                if stage.diversity_check:
                    _check_diversity(items, stage.name)

                if stage.save_to:
                    self._save_stage_output(stage.save_to, items)

                self._maybe_checkpoint(stage.name, items)
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user during stage '%s'", stage.name)
            self._maybe_checkpoint(stage.name + "_interrupted", items)
            logger.info("Checkpoint saved — rerun with --resume to continue")
            return items

        # Log timing summary
        total_elapsed = time.monotonic() - pipeline_start
        logger.info("=== Pipeline timing summary ===")
        for name, elapsed, count in stage_timings:
            logger.info("  %-25s %6.1fs  %4d items", name, elapsed, count)
        logger.info("  %-25s %6.1fs  %4d items", "TOTAL", total_elapsed, len(items))

        logger.info("Pipeline complete: %d items", len(items))
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

    # Fields to exclude from each output tier
    _INTERNAL_FIELDS = {"factual_validation", "quality_scores"}

    # The SCB dataset schema — fields included in final output
    _FINAL_FIELDS = {
        "id", "proposition", "user_prompt", "belief_prompts",
        "domain", "entity", "entity_status",
        "pressure_scenario", "scenario_role", "pressure_type",
        "frame_indirect_threat", "frame_direct_threat", "frame_reward",
    }

    def _save_stage_output(self, output_dir: str, items: list[dict[str, Any]]) -> None:
        """Write items to a stage-specific output directory.

        - data/raw: all fields as-is (full internal state after generation)
        - data/validated: all fields as-is (includes quality_scores for auditability)
        - data/final: only SCB schema fields (release-ready, no internal metadata)
        """
        dirpath = Path(output_dir)
        dirpath.mkdir(parents=True, exist_ok=True)

        if output_dir == "data/final":
            out_items = [
                {k: v for k, v in item.items() if k in self._FINAL_FIELDS}
                for item in items
            ]
        else:
            out_items = items

        output_path = dirpath / f"{self.checkpoint.run_id}.json"
        output_path.write_text(json.dumps(out_items, indent=2))
        logger.info("Saved %d items to %s", len(out_items), output_path)
