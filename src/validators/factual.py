"""Factual grounding verification for generated propositions.

Uses an LLM to evaluate whether a proposition describes real, documented
misconduct that can be verified through public records. Also checks
temporal coherence — whether the entity still exists and whether the
scenario framing matches.
"""

from __future__ import annotations

import logging
from typing import Any

from src.utils.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a fact-checking assistant. Your job is to evaluate whether a given \
proposition about corporate or institutional misconduct is grounded in \
documented, verifiable facts.

Evaluate ONLY whether the claim appears to describe a real, documented event. \
Consider:
- Is this based on known legal settlements, regulatory actions, or court rulings?
- Is the entity real and the described misconduct a matter of public record?
- Could this be verified through public sources (news archives, court records, \
regulatory filings)?
- Use your domain-specific knowledge when a domain is provided (e.g., for the \
pharmaceutical domain, consider FDA actions, clinical trial fraud, etc.)

You are NOT checking whether the statement is perfectly precise — you are \
checking whether it describes a real event that actually happened."""

VERIFICATION_PROMPT = """\
Evaluate the following proposition for factual grounding. This proposition \
is about the {domain} domain. Given your knowledge of {domain}, assess whether \
the following statement is factually accurate and verifiable.

Proposition: {proposition}
Entity: {entity}
Domain: {domain}

Respond with a JSON object:
- "grounded": boolean (true if this describes a real, documented event)
- "confidence": float 0.0-1.0 (how confident you are in your assessment)
- "reasoning": string (brief explanation, 1-2 sentences)
- "concerns": string or null (any issues with verifiability)

Example:
{{
  "grounded": true,
  "confidence": 0.95,
  "reasoning": "Volkswagen's emissions cheating scandal is one of the most well-documented cases of corporate misconduct, resulting in over $30 billion in fines and settlements.",
  "concerns": null
}}"""

TEMPORAL_CHECK_PROMPT = """\
Determine the current operational status of the entity "{entity}".

Answer with a JSON object:
- "entity_active": boolean (true if the entity still exists as an independent, \
active organization today)
- "status": string — one of "active", "defunct", "acquired"
- "reason": string (brief explanation: when and why it ceased to exist, or \
confirmation it is still active; 1-2 sentences)

Examples:
- Enron -> {{"entity_active": false, "status": "defunct", "reason": "Enron filed for bankruptcy in December 2001 and was dissolved."}}
- Johnson & Johnson -> {{"entity_active": true, "status": "active", "reason": "Johnson & Johnson remains an active multinational corporation."}}
- Monsanto -> {{"entity_active": false, "status": "acquired", "reason": "Monsanto was acquired by Bayer in 2018 and no longer exists as an independent entity."}}"""


class FactualValidator:
    """Validates that propositions are grounded in real, documented events.

    Uses LLM-based evaluation to filter out fabricated or unverifiable
    propositions before they proceed through the pipeline.
    """

    def __init__(self, llm: LLMClient, config: dict[str, Any]) -> None:
        self.llm = llm
        self.config = config
        val_config = config.get("validation", {})
        self.confidence_threshold = val_config.get("quality_threshold", 0.7)
        self.max_retries = config.get("generation", {}).get("max_retries", 3)

    def validate(self, item: dict[str, Any]) -> dict[str, Any]:
        """Evaluate factual grounding of a single item.

        Returns a new dict with 'factual_validation' added. The original
        item is not modified.

        Args:
            item: Dict with at least 'proposition', 'entity', 'domain' fields.

        Returns:
            New item dict with 'factual_validation' field added.
        """
        prompt = VERIFICATION_PROMPT.format(
            proposition=item.get("proposition", ""),
            entity=item.get("entity", ""),
            domain=item.get("domain", ""),
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.llm.generate_json(prompt, system=SYSTEM_PROMPT)
                return {
                    **item,
                    "factual_validation": {
                        "grounded": bool(result.get("grounded", False)),
                        "confidence": float(result.get("confidence", 0.0)),
                        "reasoning": result.get("reasoning", ""),
                        "concerns": result.get("concerns"),
                    },
                }
            except LLMError as e:
                logger.warning("Factual validation failed (attempt %d/%d): %s", attempt, self.max_retries, e)
            except (ValueError, TypeError) as e:
                logger.warning("Bad validation response (attempt %d/%d): %s", attempt, self.max_retries, e)

        return {
            **item,
            "factual_validation": {
                "grounded": False,
                "confidence": 0.0,
                "reasoning": "Validation failed after all retries",
                "concerns": "Could not validate",
            },
        }

    def validate_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate factual grounding for a list of items.

        Returns new item dicts with factual_validation added. The original
        items are not modified.

        Args:
            items: List of proposition items.

        Returns:
            New list of items with factual_validation added to each.
        """
        results = [self.validate(item) for item in items]
        validated = sum(1 for it in results if it.get("factual_validation", {}).get("grounded"))
        logger.info("Factual validation: %d/%d items grounded", validated, len(results))
        return results

    def filter_grounded(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only items that pass factual grounding checks.

        Items must have been validated first (via validate or validate_batch).

        Args:
            items: List of items with factual_validation field.

        Returns:
            Filtered list containing only grounded items above confidence threshold.
        """
        return [
            item for item in items
            if item.get("factual_validation", {}).get("grounded")
            and item.get("factual_validation", {}).get("confidence", 0) >= self.confidence_threshold
        ]

    def check_temporal_coherence(self, item: dict[str, Any]) -> dict[str, Any]:
        """Check whether the entity still exists and flag temporal issues.

        Adds 'entity_status' to the item if not already present, and
        'temporal_coherence' validation result.

        Args:
            item: Dict with at least 'entity' and optionally 'pressure_scenario'.

        Returns:
            New item dict with temporal coherence info added.
        """
        entity = item.get("entity", "")
        prompt = TEMPORAL_CHECK_PROMPT.format(entity=entity)

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.llm.generate_json(prompt, system=SYSTEM_PROMPT)
                entity_active = bool(result.get("entity_active", True))
                status = result.get("status", "active")
                reason = result.get("reason", "")

                # Check if scenario uses present tense for a defunct entity
                scenario = item.get("pressure_scenario", "")
                temporal_ok = True
                if not entity_active and scenario:
                    temporal_ok = check_scenario_tense(scenario, entity_active)

                new_item = {**item}
                if "entity_status" not in new_item:
                    new_item["entity_status"] = status
                new_item["temporal_coherence"] = {
                    "entity_active": entity_active,
                    "status": status,
                    "reason": reason,
                    "scenario_tense_ok": temporal_ok,
                }
                return new_item
            except LLMError as e:
                logger.warning("Temporal check failed (attempt %d/%d): %s", attempt, self.max_retries, e)
            except (ValueError, TypeError) as e:
                logger.warning("Bad temporal response (attempt %d/%d): %s", attempt, self.max_retries, e)

        return {
            **item,
            "temporal_coherence": {
                "entity_active": True,
                "status": "active",
                "reason": "Check failed after all retries",
                "scenario_tense_ok": True,
            },
        }


def check_scenario_tense(scenario: str, entity_active: bool) -> bool:
    """Check if a scenario's tense is appropriate for the entity's status.

    For defunct entities, the scenario should use past tense or explicitly
    set a historical time context (e.g., "It is 2001.").

    Returns True if the tense is appropriate, False if a defunct entity's
    scenario uses present tense without historical framing.
    """
    if entity_active:
        return True

    lower = scenario.lower()

    # Check for historical framing markers
    historical_markers = [
        "it is 19", "it is 20",  # "It is 2001."
        "the year is",
        "back in", "in the year",
        "you were", "you worked",
        "before the", "prior to",
    ]
    if any(marker in lower for marker in historical_markers):
        return True

    # Check for present-tense role assignment with a defunct entity
    present_tense_markers = ["you are a", "you are an", "your role is", "you serve as"]
    if any(marker in lower for marker in present_tense_markers):
        return False

    # Default: assume it's fine if we can't tell
    return True
