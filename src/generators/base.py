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

    def generate(self, count: int) -> list[dict[str, Any]]:
        """Generate items in batches with basic validation.

        Calls generate_batch and filters through validate_item.
        Subclasses typically don't need to override this.

        Args:
            count: Total number of valid items desired.

        Returns:
            List of validated item dicts.
        """
        batch_size = self.config.get("generation", {}).get("batch_size", 10)
        items: list[dict[str, Any]] = []

        while len(items) < count:
            needed = min(batch_size, count - len(items))
            batch = self.generate_batch(needed)
            for item in batch:
                if self.validate_item(item):
                    items.append(item)
                else:
                    logger.warning("Item failed basic validation, skipping: %s", item.get("id", "?"))

        return items[:count]
