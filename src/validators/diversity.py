"""Diversity metrics for dataset coverage analysis.

Computes distribution statistics across domains, entities, scenario
roles, and other categorical dimensions. Used to identify gaps in
coverage and guide additional generation.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class DiversityAnalyzer:
    """Analyzes coverage and diversity across dataset items.

    Computes distribution metrics for categorical fields (domain, entity,
    scenario_role) and flags under-represented categories against target
    domain lists from config.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.target_domains = config.get("generation", {}).get("domains", [])

    def analyze(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute diversity metrics for a set of items.

        Args:
            items: List of dataset items with domain, entity, scenario_role fields.

        Returns:
            Dict with distribution counts, coverage gaps, and summary stats.
        """
        domain_counts = Counter(item.get("domain", "unknown") for item in items)
        entity_counts = Counter(item.get("entity", "unknown") for item in items)
        role_counts = Counter(item.get("scenario_role", "unknown") for item in items)

        # Identify domains from config that have no items
        missing_domains = [d for d in self.target_domains if d not in domain_counts]

        # Entity concentration — flag entities that appear too often
        total = len(items) if items else 1
        entity_concentration = {
            entity: count / total
            for entity, count in entity_counts.most_common(10)
        }

        metrics = {
            "total_items": len(items),
            "domains": dict(domain_counts.most_common()),
            "unique_domains": len(domain_counts),
            "entities": dict(entity_counts.most_common()),
            "unique_entities": len(entity_counts),
            "scenario_roles": dict(role_counts.most_common()),
            "unique_roles": len(role_counts),
            "missing_domains": missing_domains,
            "entity_concentration": entity_concentration,
            "domain_coverage": len(domain_counts) / len(self.target_domains) if self.target_domains else 1.0,
        }

        logger.info(
            "Diversity: %d items, %d domains (%.0f%% coverage), %d unique entities, %d roles",
            len(items),
            len(domain_counts),
            metrics["domain_coverage"] * 100,
            len(entity_counts),
            len(role_counts),
        )
        if missing_domains:
            logger.info("Missing domains: %s", ", ".join(missing_domains))

        return metrics

    def suggest_generation(self, items: list[dict[str, Any]], target_count: int) -> dict[str, int]:
        """Suggest how many items to generate per domain to improve coverage.

        Distributes remaining items needed proportionally toward
        under-represented domains.

        Args:
            items: Current dataset items.
            target_count: Desired total number of items.

        Returns:
            Dict mapping domain names to suggested generation counts.
        """
        remaining = max(0, target_count - len(items))
        if remaining == 0:
            return {}

        domain_counts = Counter(item.get("domain", "unknown") for item in items)
        if not self.target_domains:
            return {"any": remaining}

        # Compute ideal per-domain count
        ideal_per_domain = target_count / len(self.target_domains)
        suggestions: dict[str, int] = {}

        for domain in self.target_domains:
            current = domain_counts.get(domain, 0)
            needed = max(0, int(ideal_per_domain - current))
            if needed > 0:
                suggestions[domain] = needed

        # Scale suggestions to fit within remaining budget
        total_suggested = sum(suggestions.values())
        if total_suggested > remaining and total_suggested > 0:
            scale = remaining / total_suggested
            suggestions = {d: max(1, int(n * scale)) for d, n in suggestions.items()}

        return suggestions
