"""Socratic tutoring and pedagogical scaffolding utilities for Nexus."""
from __future__ import annotations

from typing import Any, Dict

from nexus.models import ReviewResult, StudyCandidate


class SocraticTutor:
    """Pedagogical scaffolding engine for Nexus.

    Packages full source + target documents, dual-urgency rationale, and diagnostic cones
    for non-metaphorical, mechanism-level Socratic inquiry and fault isolation.
    """

    @staticmethod
    def build_study_brief(candidate: StudyCandidate) -> str:
        """Construct a complete, untruncated Socratic review brief."""
        mode_label = {
            "repair": "Surgical Bottom-Up Repair (known structural weakness)",
            "leverage": "Top-Down High-Leverage Check (refreshes decayed prerequisite cone)",
            "fresh": "Verification Check (currently fresh)",
        }.get(candidate.review_reason, candidate.review_reason)

        lines = [
            f"# Socratic Study Target: {candidate.concept.title}",
            (
                f"**Domain**: {candidate.concept.topic} | "
                f"**Strategy**: {mode_label}"
            ),
            (
                f"**Urgency**: {candidate.urgency_score:.2f} "
                f"(fault={candidate.fault_score:.2f}, leverage={candidate.decay_leverage:.2f}) | "
                f"**Live Strength**: {candidate.effective_strength:.2f} | "
                f"**Stability**: {candidate.stability_score:.2f} | "
                f"**Structural Load**: {candidate.structural_load}"
            ),
            "",
            "## Prerequisite / Concept Document",
            candidate.concept.content,
        ]

        if candidate.frayed_target and candidate.frayed_connection:
            conn = candidate.frayed_connection
            lines.extend([
                "",
                "## Target Upward Connection",
                f"- **Target Application**: {candidate.frayed_target.title} (`{candidate.frayed_target.id}`)",
                f"- **Recorded Strength**: {conn.strength:.2f} / 1.00 | **Live Effective Strength**: {candidate.effective_strength:.2f} / 1.00",
                f"- **Half-Life**: {conn.half_life:.2f} days | **Direct Reviews**: {conn.review_count}",
            ])
            if candidate.downstream_impact:
                impact_titles = ", ".join(c.title for c in candidate.downstream_impact)
                lines.append(f"- **Downstream Concepts Supported**: {impact_titles}")

            lines.extend([
                "",
                f"## Target Application Document: {candidate.frayed_target.title}",
                candidate.frayed_target.content,
            ])

            if candidate.target_prerequisites:
                lines.append("")
                lines.append("## Sibling Prerequisites Feeding Target (For Fault Isolation)")
                for p in candidate.target_prerequisites:
                    lines.append(
                        f"- **{p['title']}** (`{p['id']}`): live strength {p['effective_strength']:.2f}"
                    )
        else:
            lines.extend([
                "",
                "## Frontier Leaf Evaluation",
                (
                    f"- **Recorded Intrinsic Score**: {candidate.concept.intrinsic_score:.2f} / 1.00 | "
                    f"**Live Effective Strength**: {candidate.effective_strength:.2f} / 1.00"
                ),
                f"- **Half-Life**: {candidate.concept.half_life:.2f} days | **Direct Reviews**: {candidate.concept.review_count}",
            ])

        if candidate.implicit_refresh_targets:
            lines.append("")
            lines.append("## Foundations Implicitly Refreshed On Mastery")
            for rt in candidate.implicit_refresh_targets:
                lines.append(
                    f"- `{rt['source_id']} -> {rt['target_id']}` (depth {rt['depth']}): "
                    f"live {rt['effective_strength']:.2f} (decayed by {rt['decay']:.2f})"
                )

        if candidate.foundational_ingredients:
            lines.append("")
            lines.append("## Direct Foundational Ingredients (Drop Down Here If Student Struggles)")
            for ing in candidate.foundational_ingredients:
                lines.append(f"- **{ing.title}** (`{ing.id}`, {ing.topic})")

        lines.extend([
            "",
            "## Invariant Rules",
            "- STRICTLY NO analogies or metaphors (no rubber sheets, rivers, bowling balls, etc.).",
            "- Frame questions directly on equations, vectors/matrices, probability distributions, dimensions, loss terms, or invariants.",
            "- Probe mechanism, limiting cases, and exact failure modes when the prerequisite is altered or omitted.",
            "- Asymmetric Fault Isolation: if the student struggles, run `diagnose` on the concept and weaken ONLY the broken link(s).",
        ])
        return "\n".join(lines)

    @staticmethod
    def build_diagnostic_brief(diagnosis: Dict[str, Any]) -> str:
        """Format a top-down fault-isolation cone from engine.diagnose_concept()."""
        lines = [
            f"# Fault-Isolation Cone: {diagnosis['title']} (`{diagnosis['concept_id']}`)",
            f"**Domain**: {diagnosis['topic']} | **Stability**: {diagnosis['stability']['stability']:.2f}",
            "",
        ]

        direct = diagnosis.get("direct_prerequisites", [])
        if direct:
            lines.append("## Direct Inbound Prerequisites (Weakest First)")
            for p in direct:
                lines.append(
                    f"- **{p['title']}** (`{p['id']}`, {p['topic']}): live strength {p['effective_strength']:.2f}"
                )
            lines.append("")

        cone = diagnosis.get("supporting_cone", [])
        if cone:
            lines.append("## Full Supporting Joint Cone (Ranked by Vulnerability)")
            for j in cone:
                lines.append(
                    f"- `{j['source_id']} -> {j['target_id']}` (hops={j['hops_below']}): "
                    f"live {j['effective_strength']:.2f} (rec {j['recorded_strength']:.2f}, "
                    f"fault={j['fault']:.2f}, decay={j['decay']:.2f}, h={j['half_life_days']:.1f}d)"
                )
        else:
            lines.append("No inbound prerequisites—this is a Layer 0 foundation node.")

        return "\n".join(lines)

    @staticmethod
    def generate_joint_probe(candidate: StudyCandidate) -> str:
        """Formulate a direct, non-metaphorical opening Socratic probe."""
        if not candidate.frayed_target or not candidate.frayed_connection:
            prereq_note = ""
            if candidate.foundational_ingredients:
                names = ", ".join(i.title for i in candidate.foundational_ingredients)
                prereq_note = f" Explain explicitly how it integrates {names}."
            return (
                f"Reviewing frontier concept {candidate.concept.title} "
                f"(live mastery: {candidate.effective_strength:.2f}).\n\n"
                f"Without using analogies, derive or state the core mathematical invariant of "
                f"{candidate.concept.title}.{prereq_note} What exact failure mode arises at its boundary conditions?"
            )

        source = candidate.concept.title
        target = candidate.frayed_target.title
        return (
            f"Regarding {target}: your live strength connecting {source} into {target} is "
            f"{candidate.effective_strength:.2f} (recorded: {candidate.frayed_connection.strength:.2f}).\n\n"
            f"Without using analogies, explain precisely how {source} operates inside {target}. "
            f"Which equation, gradient, or constraint breaks if {source} is omitted or misapplied?"
        )

    @staticmethod
    def format_review_summary(result: ReviewResult) -> str:
        """Format a concise summary of a review update and any implicit reinforcements."""
        if result.mode == "joint":
            header = f"Updated joint `{result.source_id} -> {result.target_id}`"
        else:
            header = f"Updated frontier concept `{result.concept_id}`"

        lines = [
            f"{header}: {result.previous_effective_strength:.2f} -> {result.new_strength:.2f} "
            f"(half-life: {result.new_half_life:.2f}d, reviews: {result.review_count})",
            f"Concept `{result.concept_id}` stability: {result.stability['stability']:.2f}",
        ]
        if result.implicitly_reinforced:
            lines.append("Implicitly reinforced foundations:")
            for ir in result.implicitly_reinforced:
                lines.append(
                    f"  - `{ir.source_id} -> {ir.target_id}` (depth {ir.depth}): "
                    f"{ir.previous_effective_strength:.2f} -> {ir.new_strength:.2f} (h={ir.new_half_life:.2f}d)"
                )
        return "\n".join(lines)