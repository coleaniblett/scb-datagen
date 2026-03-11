"""Abstract base class for all dataset generators.

Each generator is responsible for producing one component of the dataset
(propositions, scenarios, frames, etc.). Generators receive an LLMClient
and config, and yield structured items.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Interface that all generators must implement.

    Subclasses should override `generate_batch` to produce items
    using the LLM client, and `validate_item` to perform basic
    structural checks before items enter the validation pipeline.
    """

    def __init__(self, llm: LLMClient, config: dict[str, Any]) -> None:
        """Initialize the generator.

        Args:
            llm: Configured LLM client for making generation calls.
            config: Full pipeline config dict (generator reads its own section).
        """
        self.llm = llm
        self.config = config

    @abstractmethod
    def generate_batch(self, count: int) -> list[dict[str, Any]]:
        """Generate a batch of items.

        Args:
            count: Number of items to generate in this batch.

        Returns:
            List of dicts, each representing one generated item.
        """

    def validate_item(self, item: dict[str, Any]) -> bool:
        """Basic structural validation of a generated item.

        Override to add generator-specific checks. This is *not* the
        full quality validation (that lives in src/validators/), just
        a quick sanity check that the item has required fields.

        Args:
            item: A single generated item dict.

        Returns:
            True if the item passes basic checks.
        """
        return True

    def generate(self, count: int, max_attempts: int = 10) -> list[dict[str, Any]]:
        """Generate items in batches with basic validation.

        Calls generate_batch and filters through validate_item.
        Subclasses typically don't need to override this.

        Args:
            count: Total number of valid items desired.
            max_attempts: Maximum consecutive empty batches before raising RuntimeError.

        Returns:
            List of validated item dicts.

        Raises:
            RuntimeError: If max_attempts consecutive calls to generate_batch
                return zero valid items.
        """
        batch_size = self.config.get("generation", {}).get("batch_size", 10)
        max_rounds = self.config.get("generation", {}).get("max_rounds", 10)
        items: list[dict[str, Any]] = []

        rounds = 0
        consecutive_empty = 0
        while len(items) < count and rounds < max_rounds:
            rounds += 1
            needed = min(batch_size, count - len(items))
            batch = self.generate_batch(needed)
            if not batch:
                consecutive_empty += 1
                logger.warning(
                    "Empty batch returned (round %d/%d, %d consecutive empty)",
                    rounds, max_rounds, consecutive_empty,
                )
                if consecutive_empty >= max_attempts:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: {max_attempts} consecutive empty batches. "
                        f"Collected {len(items)}/{count} items. "
                        f"Check LLM connectivity and prompt configuration."
                    )
                continue
            consecutive_empty = 0
            for item in batch:
                if self.validate_item(item):
                    items.append(item)
                else:
                    logger.warning("Item failed basic validation, skipping: %s", item.get("id", "?"))

        if len(items) < count:
            logger.warning(
                "Only generated %d/%d valid items after %d rounds", len(items), count, max_rounds
            )

        return items[:count]
