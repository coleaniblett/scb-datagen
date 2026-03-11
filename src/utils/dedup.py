"""Semantic deduplication for dataset items.

Detects near-duplicate propositions using entity matching and text
similarity. Keeps dependencies minimal by using difflib for string
similarity rather than requiring embedding models.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.8


def deduplicate(
    items: list[dict[str, Any]],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Remove near-duplicate items based on proposition text similarity.

    Uses SequenceMatcher for pairwise comparison. Items that are too similar
    to an already-accepted item are dropped. Comparison is O(n^2) but
    acceptable for dataset sizes up to a few thousand items.

    TODO: For scale beyond ~2k items, consider switching to MinHash via
    ``datasketch`` for approximate O(n) dedup.

    Args:
        items: List of dataset items with 'proposition' field.
        threshold: Similarity ratio above which items are considered duplicates.
                   Default 0.8 (80% similar text).

    Returns:
        Deduplicated list preserving original order (first occurrence kept).
    """
    if not items:
        return []

    kept: list[dict[str, Any]] = []
    dropped_ids: list[str] = []

    for item in items:
        prop = item.get("proposition", "")
        is_dup = False

        for accepted in kept:
            ratio = SequenceMatcher(None, prop.lower(), accepted.get("proposition", "").lower()).ratio()
            if ratio >= threshold:
                is_dup = True
                dropped_ids.append(item.get("id", "?"))
                break

        if not is_dup:
            kept.append(item)

    if dropped_ids:
        logger.info("Dedup: removed %d duplicates (threshold=%.2f): %s", len(dropped_ids), threshold, dropped_ids)
    else:
        logger.info("Dedup: no duplicates found (threshold=%.2f)", threshold)

    return kept


def find_duplicates(
    items: list[dict[str, Any]],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Find all pairs of near-duplicate items.

    Useful for inspection before committing to dedup.

    Args:
        items: List of dataset items with 'proposition' field.
        threshold: Similarity threshold.

    Returns:
        List of (id_a, id_b, similarity) tuples for all duplicate pairs.
    """
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            ratio = SequenceMatcher(
                None,
                a.get("proposition", "").lower(),
                b.get("proposition", "").lower(),
            ).ratio()
            if ratio >= threshold:
                pairs.append((a.get("id", "?"), b.get("id", "?"), ratio))
    return pairs
