"""Data models for Nexus: Directed Cognitive Blueprint & Connection-Strength Engine."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConceptBlueprint(BaseModel):
    """A rich personal blueprint of a concept's mental model."""

    id: str = Field(..., description="Unique normalized concept slug, e.g. 'bayes-theorem'")
    title: str = Field(..., description="Concept title")
    topic: str = Field(default="General", description="Knowledge domain or category")
    mental_model: str = Field(
        ...,
        description="Comprehensive explanation of the core mechanism that clicked for the student",
    )
    anchor_example: str = Field(
        default="",
        description="Specific intuitive example, derivation step, or concrete scenario that unlocked it",
    )
    pitfalls: str = Field(
        default="",
        description="Resolved misconceptions, confusions overcome, or what this concept is NOT",
    )
    intrinsic_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Baseline mastery score (primarily for frontier leaf nodes with 0 outgoing links)",
    )
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


class Connection(BaseModel):
    """A directed upward connection from prerequisite to higher application."""

    source_id: str = Field(..., description="Prerequisite concept slug")
    target_id: str = Field(..., description="Higher-level application concept slug (pointing UP)")
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Connection strength in [0.0, 1.0]",
    )
    last_reviewed: Optional[str] = Field(default=None)
    review_count: int = Field(default=0, ge=0)


class StudyCandidate(BaseModel):
    """A prioritized concept surfaced for Socratic review, packaged with its context bundle."""

    concept: ConceptBlueprint
    urgency_score: float = Field(..., description="Higher = more urgently needs review")
    stability_score: float = Field(..., description="Overall concept stability across all outgoing uses")
    frayed_target: Optional[ConceptBlueprint] = Field(
        default=None,
        description="The higher-level concept where this concept's connection is weakest",
    )
    frayed_connection: Optional[Connection] = Field(
        default=None,
        description="The specific weak upward connection being repaired",
    )
    foundational_ingredients: List[ConceptBlueprint] = Field(
        default_factory=list,
        description="Inbound prerequisite blueprints needed to reconstruct this concept from the ground up",
    )
