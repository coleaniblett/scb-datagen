"""Pydantic models for pipeline item validation at stage boundaries.

Defines the expected schema at each pipeline stage, progressively adding
fields as items flow through generation and enrichment. Using
``extra="forbid"`` catches field-name typos immediately.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


def _check_non_blank(v: str, field_name: str) -> str:
    """Validate that a string field is not blank after stripping whitespace."""
    if not v or not v.strip():
        raise ValueError(f"{field_name} must not be blank")
    return v


VALID_ENTITY_STATUSES = {"active", "defunct", "acquired"}
VALID_PRESSURE_TYPES = {"role_based", "directive_based", "contextual"}


class PropositionItem(BaseModel):
    """Schema after proposition generation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    proposition: str
    domain: str
    entity: str
    entity_status: Literal["active", "defunct", "acquired"]
    user_prompt: str
    belief_prompts: list[str]

    @field_validator("id", "proposition", "domain", "entity", "user_prompt")
    @classmethod
    def must_be_non_blank(cls, v: str, info) -> str:
        return _check_non_blank(v, info.field_name)

    @field_validator("belief_prompts")
    @classmethod
    def at_least_two_belief_prompts(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("belief_prompts must have at least 2 entries")
        for i, prompt in enumerate(v):
            if not prompt or not prompt.strip():
                raise ValueError(f"belief_prompts[{i}] must not be blank")
        return v


class ScenarioItem(PropositionItem):
    """Schema after scenario enrichment."""

    pressure_scenario: str
    scenario_role: str
    pressure_type: Literal["role_based", "directive_based", "contextual"]

    @field_validator("pressure_scenario", "scenario_role")
    @classmethod
    def scenario_fields_non_blank(cls, v: str, info) -> str:
        return _check_non_blank(v, info.field_name)


class FramedItem(ScenarioItem):
    """Schema after frame enrichment."""

    frame_indirect_threat: str
    frame_direct_threat: str
    frame_reward: str

    @field_validator("frame_indirect_threat", "frame_direct_threat", "frame_reward")
    @classmethod
    def frame_fields_non_blank(cls, v: str, info) -> str:
        return _check_non_blank(v, info.field_name)
