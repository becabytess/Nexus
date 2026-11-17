"""Core graph engine for Nexus: directed upward dependencies, connection strengths, and Socratic retrieval."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from nexus.models import (
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

    The only thing creating the hierarchy is the connection strength w in [0.0, 1.0].
    Every concept is an identical, full-fidelity ConceptBlueprint.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptBlueprint] = {}
        self.connections: List[Connection] = []
        self._adj_up: Dict[str, List[Connection]] = defaultdict(list)    # source_id -> outgoing connections
        self._adj_down: Dict[str, List[Connection]] = defaultdict(list)  # target_id -> incoming connections

    def _index_connection(self, conn: Connection) -> None:
        """Register a connection in both the upward and downward adjacency indices."""
        up_list = self._adj_up[conn.source_id]
        if conn not in up_list:
            up_list.append(conn)

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
    ) -> Connection:
        """Create or update a directed connection between prerequisite and higher concept."""
        if source_id not in self.concepts:
            raise KeyError(f"Source concept {source_id!r} not found in graph")
        if target_id not in self.concepts:
            raise KeyError(f"Target concept {target_id!r} not found in graph")

        clamped = max(0.0, min(1.0, float(strength)))

        # Check if connection already exists in upward index
        for conn in self._adj_up.get(source_id, []):
            if conn.target_id == target_id:
                conn.strength = clamped
                return conn

        conn = Connection(
            source_id=source_id,
            target_id=target_id,
            strength=clamped,
        )
        self.connections.append(conn)
        self._index_connection(conn)
        return conn

    def add_bundle(
        self,
        concept: ConceptBlueprint,
        prerequisites: List[ConceptBlueprint],
    ) -> Dict[str, Any]:
        """Record a learning session: adds the main concept, adds all prerequisites,

        and wires connections pointing UP from each prerequisite into the main concept.
        Every concept is stored as an identical full ConceptBlueprint.
        """
        # 1. Save the main concept
        self.add_concept(concept)

        saved_prereqs: List[ConceptBlueprint] = []
        created_connections: List[Connection] = []

        # 2. Process each prerequisite concept
        for p in prerequisites:
            pid = p.id or slugify(p.title)
            p.id = pid
            if pid in self.concepts:
                existing = self.concepts[pid]
                if p.mental_model and len(p.mental_model) > len(existing.mental_model):
                    existing.mental_model = p.mental_model
                if p.anchor_example and not existing.anchor_example:
                    existing.anchor_example = p.anchor_example
                if p.pitfalls and not existing.pitfalls:
                    existing.pitfalls = p.pitfalls
                bp = self.add_concept(existing)
            else:
                bp = self.add_concept(p)

            saved_prereqs.append(bp)

            # 3. Wire upward connection: prerequisite ---> main concept
            conn = self.connect(
                source_id=bp.id,
                target_id=concept.id,
                strength=1.0,
            )
            created_connections.append(conn)

        return {
            "concept": concept,
            "prerequisites": saved_prereqs,
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
        """Look DOWN: retrieve all component connections of a parent concept."""
        return self.get_incoming(concept_id)

    def get_applications(self, concept_id: str) -> List[Connection]:
        """Look UP: retrieve all higher-level application connections of a prerequisite."""
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
        """
        candidates: List[StudyCandidate] = []

        for cid, concept in self.concepts.items():
            outgoing = self.get_outgoing(cid)
            metrics = self.calculate_concept_stability(cid)

            if outgoing:
                sorted_links = sorted(outgoing, key=lambda c: c.strength)
                weakest_link = sorted_links[0]
                urgency = (1.0 - weakest_link.strength) * (1.0 + 0.3 * math.log(1.0 + len(outgoing)))
                target_concept = self.concepts.get(weakest_link.target_id)
                ingredients = self.get_foundational_ingredients(cid)

                candidates.append(
                    StudyCandidate(
                        concept=concept,
                        urgency_score=round(urgency, 4),
                        stability_score=metrics["stability"],
                        frayed_target=target_concept,
                        frayed_connection=weakest_link,
                        foundational_ingredients=ingredients,
                    )
                )
            else:
                urgency = 1.0 - concept.intrinsic_score
                ingredients = self.get_foundational_ingredients(cid)
                candidates.append(
                    StudyCandidate(
                        concept=concept,
                        urgency_score=round(urgency, 4),
                        stability_score=metrics["stability"],
                        frayed_target=None,
                        frayed_connection=None,
                        foundational_ingredients=ingredients,
                    )
                )

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
        """Update the health of a specific directed connection (prerequisite -> target)."""
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
