"""Core graph engine for Nexus: directed upward dependencies, connection strengths, and Socratic retrieval."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from nexus.models import (
    CheckpointSpec,
    ConceptBlueprint,
    Connection,
    StudyCandidate,
    _now_iso,
)


def slugify(text: str) -> str:
    """Normalize a title string into a clean lowercase identifier."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", cleaned)


class NexusEngine:
    """Directed Cognitive Blueprint & Connection-Strength Knowledge Engine.

    Maintains bidirectional adjacency pointers for instant O(1) traversal in both directions:
        - UPWARD (_adj_up): [Prerequisite] ──(w)──► [Higher Application] (where concept is applied)
        - DOWNWARD (_adj_down): [Higher Application] ◄──(w)── [Prerequisite] (what concept is composed of)

    Selection scans UPWARD: concepts with weak outgoing connections into higher applications
    are prioritized for study, packaged together with their foundational ingredients retrieved
    via instant DOWNWARD pointers.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptBlueprint] = {}
        self.connections: List[Connection] = []
        # Bidirectional adjacency indices for O(1) navigation
        self._adj_up: Dict[str, List[Connection]] = defaultdict(list)    # source_id -> outgoing connections
        self._adj_down: Dict[str, List[Connection]] = defaultdict(list)  # target_id -> incoming connections

    def _index_connection(self, conn: Connection) -> None:
        """Register a connection in both the upward and downward adjacency indices."""
        # Update or append in _adj_up
        up_list = self._adj_up[conn.source_id]
        if conn not in up_list:
            up_list.append(conn)

        # Update or append in _adj_down
        down_list = self._adj_down[conn.target_id]
        if conn not in down_list:
            down_list.append(conn)

    def reindex(self) -> None:
        """Rebuild bidirectional adjacency indices from the connection pool."""
        self._adj_up.clear()
        self._adj_down.clear()
        for conn in self.connections:
            self._index_connection(conn)

    # -------------------------------------------------------------------------
    # Graph Construction & Modification
    # -------------------------------------------------------------------------

    def add_concept(self, blueprint: ConceptBlueprint) -> ConceptBlueprint:
        """Register or update a rich concept blueprint."""
        blueprint.updated_at = _now_iso()
        self.concepts[blueprint.id] = blueprint
        return blueprint

    def connect(
        self,
        source_id: str,
        target_id: str,
        strength: float = 1.0,
        context_role: str = "",
    ) -> Connection:
        """Create or update a directed connection between prerequisite and higher concept.

        Maintains instant pointers in both directions:
        - source_id registers an UPWARD pointer to target_id.
        - target_id registers a DOWNWARD pointer to source_id.
        """
        if source_id not in self.concepts:
            raise KeyError(f"Source concept {source_id!r} not found in graph")
        if target_id not in self.concepts:
            raise KeyError(f"Target concept {target_id!r} not found in graph")

        clamped = max(0.0, min(1.0, float(strength)))

        # Check if connection already exists in upward index
        for conn in self._adj_up.get(source_id, []):
            if conn.target_id == target_id:
                conn.strength = clamped
                if context_role:
                    conn.context_role = context_role
                return conn

        conn = Connection(
            source_id=source_id,
            target_id=target_id,
            strength=clamped,
            context_role=context_role,
        )
        self.connections.append(conn)
        self._index_connection(conn)
        return conn

    def add_bundle(
        self,
        target_blueprint: ConceptBlueprint,
        checkpoints: List[CheckpointSpec],
    ) -> Dict[str, Any]:
        """Record an 'aha!' learning session: creates the main concept, creates/updates checkpoints,

        and wires directed connections pointing UP from each checkpoint into the main concept.
        """
        # 1. Save the main concept
        self.add_concept(target_blueprint)

        created_checkpoints: List[ConceptBlueprint] = []
        created_connections: List[Connection] = []

        # 2. Process each checkpoint
        for cp in checkpoints:
            cp_id = slugify(cp.title)
            # Re-use existing concept if present, updating if new details are provided
            if cp_id in self.concepts:
                existing = self.concepts[cp_id]
                if cp.mental_model and len(cp.mental_model) > len(existing.mental_model):
                    existing.mental_model = cp.mental_model
                if cp.anchor_example and not existing.anchor_example:
                    existing.anchor_example = cp.anchor_example
                if cp.pitfalls and not existing.pitfalls:
                    existing.pitfalls = cp.pitfalls
                blueprint = self.add_concept(existing)
            else:
                blueprint = ConceptBlueprint(
                    id=cp_id,
                    title=cp.title,
                    topic=cp.topic or target_blueprint.topic,
                    mental_model=cp.mental_model,
                    anchor_example=cp.anchor_example,
                    pitfalls=cp.pitfalls,
                    intrinsic_score=1.0,
                )
                self.add_concept(blueprint)

            created_checkpoints.append(blueprint)

            # 3. Wire upward connection: checkpoint ---> main concept
            conn = self.connect(
                source_id=blueprint.id,
                target_id=target_blueprint.id,
                strength=1.0,
                context_role=cp.context_role,
            )
            created_connections.append(conn)

        return {
            "target": target_blueprint,
            "checkpoints": created_checkpoints,
            "connections": created_connections,
        }

    # -------------------------------------------------------------------------
    # Graph Inspection & Neighborhoods (O(1) Bidirectional Traversal)
    # -------------------------------------------------------------------------

    def get_outgoing(self, concept_id: str) -> List[Connection]:
        """Look UP: instant O(1) return of all upward connections where concept is used as an ingredient."""
        return list(self._adj_up.get(concept_id, []))

    def get_incoming(self, concept_id: str) -> List[Connection]:
        """Look DOWN: instant O(1) return of all downward connections pointing into this concept's ingredients."""
        return list(self._adj_down.get(concept_id, []))

    def get_components(self, concept_id: str) -> List[Connection]:
        """Look DOWN: explicit top-down query retrieving all component ingredients of a parent concept."""
        return self.get_incoming(concept_id)

    def get_applications(self, concept_id: str) -> List[Connection]:
        """Look UP: explicit bottom-up query retrieving all higher-level applications of a prerequisite."""
        return self.get_outgoing(concept_id)

    def get_foundational_ingredients(self, concept_id: str) -> List[ConceptBlueprint]:
        """Retrieve all component blueprints directly needed to construct this concept from below."""
        inbound = self.get_incoming(concept_id)
        return [self.concepts[c.source_id] for c in inbound if c.source_id in self.concepts]

    def get_higher_applications(self, concept_id: str) -> List[ConceptBlueprint]:
        """Retrieve all higher blueprints that this concept points UP into."""
        outbound = self.get_outgoing(concept_id)
        return [self.concepts[c.target_id] for c in outbound if c.target_id in self.concepts]

    # -------------------------------------------------------------------------
    # Stability & Upward Study Selection
    # -------------------------------------------------------------------------

    def calculate_concept_stability(self, concept_id: str) -> Dict[str, Any]:
        """Calculate concept stability based on its outgoing connections (Quality * Breadth).

        Stability = (Average Outgoing Connection Strength) * (1.0 + 0.5 * ln(1 + K))
        Leaves at the top (K = 0) fall back to their intrinsic_score baseline.
        """
        concept = self.concepts.get(concept_id)
        if not concept:
            raise KeyError(f"Concept {concept_id!r} not found")

        outgoing = self.get_outgoing(concept_id)
        k = len(outgoing)

        if k == 0:
            # Frontier leaf node with no higher-level concepts yet
            return {
                "concept_id": concept_id,
                "breadth_k": 0,
                "avg_strength": float(concept.intrinsic_score),
                "min_strength": float(concept.intrinsic_score),
                "breadth_bonus": 1.0,
                "stability": float(concept.intrinsic_score),
                "is_frontier_leaf": True,
            }

        strengths = [c.strength for c in outgoing]
        avg_s = float(sum(strengths) / k)
        min_s = float(min(strengths))
        # Diminishing return breadth multiplier: 1.0 at K=0, 1.35 at K=1, 1.69 at K=3, 2.04 at K=7
        breadth_bonus = 1.0 + 0.5 * math.log(1.0 + k)
        stability = avg_s * breadth_bonus

        return {
            "concept_id": concept_id,
            "breadth_k": k,
            "avg_strength": round(avg_s, 4),
            "min_strength": round(min_s, 4),
            "breadth_bonus": round(breadth_bonus, 4),
            "stability": round(stability, 4),
            "is_frontier_leaf": False,
        }

    def get_study_candidates(self, top_k: int = 3) -> List[StudyCandidate]:
        """Surface concepts most urgently in need of Socratic study by looking UPWARD.

        Scans for frayed connections (connections with lowest strength w < 0.95),
        prioritizing foundational concepts whose weakness compromises higher structures.
        Each candidate is packaged with:
        - The weak concept blueprint.
        - The specific higher-level target where the connection is frayed.
        - The foundational children needed to rebuild the concept from below.
        """
        candidates: List[StudyCandidate] = []

        # 1. Evaluate every concept with outgoing connections
        for cid, concept in self.concepts.items():
            outgoing = self.get_outgoing(cid)
            metrics = self.calculate_concept_stability(cid)

            if outgoing:
                # Find the weakest upward connection for this concept
                sorted_links = sorted(outgoing, key=lambda c: c.strength)
                weakest_link = sorted_links[0]

                # Urgency is driven by the deficit on the weakest link
                # (1.0 - strength) boosted by foundational breadth (number of things relying on it)
                urgency = (1.0 - weakest_link.strength) * (1.0 + 0.3 * math.log(1.0 + len(outgoing)))

                target_concept = self.concepts.get(weakest_link.target_id)
                ingredients = self.get_foundational_ingredients(cid)

                reason = (
                    f"Frayed connection ({weakest_link.strength:.2f}) into {target_concept.title if target_concept else weakest_link.target_id}. "
                    f"Context: {weakest_link.context_role or 'prerequisite role'}."
                )

                candidates.append(
                    StudyCandidate(
                        concept=concept,
                        urgency_score=round(urgency, 4),
                        stability_score=metrics["stability"],
                        frayed_target=target_concept,
                        frayed_connection=weakest_link,
                        foundational_ingredients=ingredients,
                        reason=reason,
                    )
                )
            else:
                # Frontier leaf node with 0 outgoing connections
                urgency = 1.0 - concept.intrinsic_score
                ingredients = self.get_foundational_ingredients(cid)
                reason = "Frontier leaf concept with 0 higher-level outgoing applications yet."
                candidates.append(
                    StudyCandidate(
                        concept=concept,
                        urgency_score=round(urgency, 4),
                        stability_score=metrics["stability"],
                        frayed_target=None,
                        frayed_connection=None,
                        foundational_ingredients=ingredients,
                        reason=reason,
                    )
                )

        # Sort by urgency descending
        candidates.sort(key=lambda c: c.urgency_score, reverse=True)
        return candidates[:top_k]

    # -------------------------------------------------------------------------
    # Socratic Review Updates
    # -------------------------------------------------------------------------

    def review_joint(
        self,
        source_id: str,
        target_id: str,
        score: float,
    ) -> Connection:
        """Update the health of a specific directed connection (prerequisite -> target).

        If score >= 0.8: connection strengthens.
        If score < 0.8: connection weakens (frayed wire identified in Socratic chat).
        """
        clamped = max(0.0, min(1.0, float(score)))
        conn = None
        for c in self.connections:
            if c.source_id == source_id and c.target_id == target_id:
                conn = c
                break

        if not conn:
            raise KeyError(f"Connection {source_id} -> {target_id} does not exist")

        conn.strength = round(clamped, 4)
        conn.last_reviewed = _now_iso()
        conn.review_count += 1
        return conn

    def review_bundle(
        self,
        target_id: str,
        component_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Apply Socratic evaluation results across the inbound joints feeding a target concept."""
        if target_id not in self.concepts:
            raise KeyError(f"Target concept {target_id!r} not found")

        updated_connections = []
        for src_id, score in component_scores.items():
            try:
                conn = self.review_joint(src_id, target_id, score)
                updated_connections.append(conn)
            except KeyError:
                continue

        return {
            "target_id": target_id,
            "target_title": self.concepts[target_id].title,
            "updated_joints": len(updated_connections),
            "joints": [
                {
                    "source_id": c.source_id,
                    "target_id": c.target_id,
                    "new_strength": c.strength,
                }
                for c in updated_connections
            ],
        }
