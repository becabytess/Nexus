"""Data models for Nexus: Directed Cognitive Blueprint & Connection-Strength Engine."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _now_iso(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_iso(timestamp: str) -> datetime:
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ConceptBlueprint(BaseModel):
    """A concept node containing a complete, standalone Markdown explainer document."""

    id: str = Field(..., min_length=1, description="Unique normalized concept slug, e.g. 'bayes-theorem'")
    title: str = Field(..., min_length=1, description="Concept title")
    topic: str = Field(default="General", description="Knowledge domain or category")
    content: str = Field(
        ...,
        description="Complete standalone Markdown explainer with derivations, LaTeX equations, invariants, and edge cases.",
    )
    intrinsic_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Baseline mastery score (used for frontier leaf nodes with 0 outgoing links)",
    )
    half_life: float = Field(
        default=1.0,
        gt=0.0,
        description="Memory half-life in days when evaluated as a frontier leaf",
    )
    last_reviewed: Optional[str] = Field(default=None)
    review_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


class Connection(BaseModel):
    """A directed upward connection from prerequisite to higher application."""

    source_id: str = Field(..., min_length=1, description="Prerequisite concept slug")
    target_id: str = Field(..., min_length=1, description="Higher-level application concept slug (pointing UP)")
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Recorded connection strength in [0.0, 1.0]",
    )
    half_life: float = Field(
        default=1.0,
        gt=0.0,
        description="Memory half-life in days before structural breadth anchoring",
    )
    created_at: str = Field(default_factory=_now_iso)
    last_reviewed: Optional[str] = Field(default=None)
    review_count: int = Field(default=0, ge=0)


class ImplicitReinforcement(BaseModel):
    """Record of fractional implicit confirmation propagated downward to a supporting joint."""

    source_id: str
    target_id: str
    depth: int = Field(..., ge=1)
    previous_effective_strength: float
    new_strength: float
    new_half_life: float


class ReviewResult(BaseModel):
    """Unified output of reviewing either a directed joint or a frontier concept."""

    mode: str = Field(..., description="'joint' or 'concept'")
    concept_id: str
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    previous_effective_strength: float
    new_strength: float
    new_half_life: float
    review_count: int
    last_reviewed: str
    stability: Dict[str, Any]
    implicitly_reinforced: List[ImplicitReinforcement] = Field(default_factory=list)


class StudyCandidate(BaseModel):
    """A prioritized joint or frontier leaf surfaced for Socratic review with full context."""

    concept: ConceptBlueprint
    urgency_score: float = Field(..., description="Total priority = fault_score + decay_leverage")
    fault_score: float = Field(default=0.0, description="Bottom-up structural repair priority from known weakness")
    decay_leverage: float = Field(default=0.0, description="Top-down implicit-confirmation mastery recoverable via time decay")
    review_reason: str = Field(default="repair", description="'repair', 'leverage', or 'fresh'")
    stability_score: float = Field(..., description="Live concept stability across outgoing uses")
    effective_strength: float = Field(..., description="Current time-decayed strength of the target joint or leaf")
    structural_load: int = Field(default=0, description="Number of downstream concepts relying on this link")
    frayed_target: Optional[ConceptBlueprint] = Field(
        default=None,
        description="Higher-level concept for a joint review (None for frontier leaves)",
    )
    frayed_connection: Optional[Connection] = Field(
        default=None,
        description="Specific upward connection under review (None for frontier leaves)",
    )
    downstream_impact: List[ConceptBlueprint] = Field(
        default_factory=list,
        description="Higher-level concepts directly or transitively supported by this joint",
    )
    foundational_ingredients: List[ConceptBlueprint] = Field(
        default_factory=list,
        description="Direct inbound prerequisite blueprints of the concept under review",
    )
    target_prerequisites: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Inbound prerequisites feeding frayed_target (for asymmetric fault isolation)",
    )
    implicit_refresh_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Decayed upstream joints that will receive implicit confirmation if this review succeeds",
    )