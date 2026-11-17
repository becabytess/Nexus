"""Socratic tutoring and pedagogical scaffolding utilities for Nexus."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from nexus.models import ConceptBlueprint, Connection, StudyCandidate


class SocraticTutor:
    """Pedagogical scaffolding engine for Nexus.

    Generates mechanism-level Socratic inquiries focused on weak upward connections,
    linking foundational ingredients into higher-level applications without metaphors.
    """

    @staticmethod
    def build_study_brief(candidate: StudyCandidate) -> str:
        """Construct a structured Socratic review brief for an agent or student."""
        lines = []
        lines.append(f"# Socratic Study Target: {candidate.concept.title}")
        lines.append(f"**Domain**: {candidate.concept.topic} | **Urgency**: {candidate.urgency_score:.2f} | **Stability**: {candidate.stability_score:.2f}\n")

        lines.append("## Concept Blueprint")
        lines.append(f"- **Mechanism**: {candidate.concept.mental_model}")
        if candidate.concept.anchor_example:
            lines.append(f"- **Anchor Example**: {candidate.concept.anchor_example}")
        if candidate.concept.pitfalls:
            lines.append(f"- **Pitfalls / Anti-patterns**: {candidate.concept.pitfalls}")

        if candidate.frayed_target and candidate.frayed_connection:
            lines.append("\n## Weak Upward Connection")
            lines.append(f"- **Target Application**: {candidate.frayed_target.title}")
            lines.append(f"- **Connection Strength**: {candidate.frayed_connection.strength:.2f} / 1.00")
            lines.append(f"- **Target Mechanism**: {candidate.frayed_target.mental_model}")

        if candidate.foundational_ingredients:
            lines.append("\n## Foundational Ingredients")
            for ing in candidate.foundational_ingredients:
                lines.append(f"- **{ing.title}** ({ing.topic}): {ing.mental_model}")

        lines.append("\n## Invariant Rules")
        lines.append("- STRICTLY NO analogies or metaphors (no rubber sheets, rivers, bowling balls, etc.).")
        lines.append("- Formulate the inquiry directly on mathematical objects, structural invariants, dimensions, or loss objectives.")
        lines.append("- Probe why and how this concept operates inside the higher-level target.")

        return "\n".join(lines)

    @staticmethod
    def generate_joint_probe(candidate: StudyCandidate) -> str:
        """Formulate a direct, non-metaphorical Socratic inquiry probing the weak joint."""
        if not candidate.frayed_target or not candidate.frayed_connection:
            return (
                f"Explain the core structural invariant of {candidate.concept.title}: "
                f"what exact mathematical or computational problem does it solve, and under what condition does it fail?"
            )

        source = candidate.concept.title
        target = candidate.frayed_target.title

        return (
            f"Regarding {target}: your recorded understanding of {source} in relation to {target} is weak ({candidate.frayed_connection.strength:.2f}).\n\n"
            f"Without using analogies, explain how {source} functions in {target}. "
            f"What specific failure mode or mathematical impossibility arises if {source} is omitted?"
        )
