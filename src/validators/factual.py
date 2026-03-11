"""Factual grounding verification for generated propositions.

Uses an LLM to evaluate whether a proposition describes real, documented
misconduct that can be verified through public records. Acts as a first
filter before more expensive quality validation.
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

You are NOT checking whether the statement is perfectly precise — you are \
checking whether it describes a real event that actually happened."""

VERIFICATION_PROMPT = """\
Evaluate the following proposition for factual grounding.

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

        Args:
            item: Dict with at least 'proposition', 'entity', 'domain' fields.

        Returns:
            The item dict with 'factual_validation' field added, containing
            grounded, confidence, reasoning, and concerns.
        """
        prompt = VERIFICATION_PROMPT.format(
            proposition=item.get("proposition", ""),
            entity=item.get("entity", ""),
            domain=item.get("domain", ""),
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.llm.generate_json(prompt, system=SYSTEM_PROMPT)
                item["factual_validation"] = {
                    "grounded": bool(result.get("grounded", False)),
                    "confidence": float(result.get("confidence", 0.0)),
                    "reasoning": result.get("reasoning", ""),
                    "concerns": result.get("concerns"),
                }
                return item
            except LLMError as e:
                logger.warning("Factual validation failed (attempt %d/%d): %s", attempt, self.max_retries, e)
            except (ValueError, TypeError) as e:
                logger.warning("Bad validation response (attempt %d/%d): %s", attempt, self.max_retries, e)

        item["factual_validation"] = {
            "grounded": False,
            "confidence": 0.0,
            "reasoning": "Validation failed after all retries",
            "concerns": "Could not validate",
        }
        return item

    def validate_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate factual grounding for a list of items.

        Args:
            items: List of proposition items.

        Returns:
            The same list with factual_validation added to each item.
        """
        for item in items:
            self.validate(item)
        validated = sum(1 for it in items if it.get("factual_validation", {}).get("grounded"))
        logger.info("Factual validation: %d/%d items grounded", validated, len(items))
        return items

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
