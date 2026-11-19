"""Core graph engine for Nexus: upward DAG, continuous decay, dual urgency, and asymmetric FIRe."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from nexus.models import (
    ConceptBlueprint,
    Connection,
    ImplicitReinforcement,
    ReviewResult,
    StudyCandidate,
    _now_iso,
    _parse_iso,
)

MIN_HALF_LIFE_DAYS = 0.25
IMPLICIT_DAMPING = 0.5
MIN_IMPLICIT_CREDIT = 0.05


def slugify(text: str) -> str:
    """Normalize a title string into a clean lowercase slug identifier."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[-\s_]+", "-", cleaned).strip("-")
    if not slug:
        raise ValueError(f"Cannot generate valid slug from {text!r}")
    return slug


def breadth_multiplier(count: int) -> float:
    """Concave structural multiplier B(n) = 1 + 0.5 * ln(1 + n).

    Shared across stability, breadth-anchored half-life, and structural-load fault priority.
    """
    return 1.0 + 0.5 * math.log(1.0 + max(0, count))


def decay_factor(elapsed_days: float, half_life: float) -> float:
    """Continuous exponential memory retention R = 2^(-dt / h) in (0, 1]."""
    dt = max(0.0, float(elapsed_days))
    hl = max(0.01, float(half_life))
    return 2.0 ** (-dt / hl)


def update_half_life(
    half_life: float,
    score: float,
    retrievability: float,
    weight: float = 1.0,
) -> float:
    """Smooth, threshold-free spacing update for half-life.

    Let delta = 2*score - 1 in [-1, 1]:
    - Passing (delta >= 0): h' = h * 2^(weight * 2 * delta * (1 - R))
      Cramming at R=1 yields zero free inflation; recall at half-life (R=0.5) doubles h.
    - Struggling (delta < 0): h' = max(0.25, h * 2^(weight * delta))
      Immediate contraction regardless of elapsed time.
    """
    s = max(0.0, min(1.0, float(score)))
    r = max(0.0, min(1.0, float(retrievability)))
    w = max(0.0, float(weight))
    delta = 2.0 * s - 1.0

    if delta >= 0.0:
        updated = half_life * (2.0 ** (w * 2.0 * delta * (1.0 - r)))
    else:
        updated = half_life * (2.0 ** (w * delta))

    return round(max(MIN_HALF_LIFE_DAYS, updated), 4)


def _resolve_now(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


class NexusEngine:
    """Directed Cognitive Blueprint & Connection-Strength Knowledge Engine.

    Maintains bidirectional adjacency indices for O(1) traversal:
        - UPWARD   (_adj_up):   [Prerequisite] --(w)--> [Higher Application]
        - DOWNWARD (_adj_down): [Higher Application] <--(w)-- [Prerequisite]
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptBlueprint] = {}
        self.connections: List[Connection] = []
        self._adj_up: Dict[str, List[Connection]] = defaultdict(list)
        self._adj_down: Dict[str, List[Connection]] = defaultdict(list)

    def _index_connection(self, conn: Connection) -> None:
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
        """Register or update a concept blueprint, preserving created_at if updating."""
        now_iso = _now_iso()
        if blueprint.id in self.concepts:
            blueprint.created_at = self.concepts[blueprint.id].created_at
        blueprint.updated_at = now_iso
        self.concepts[blueprint.id] = blueprint
        return blueprint

    def get_connection(self, source_id: str, target_id: str) -> Optional[Connection]:
        """O(1) lookup for a directed connection (source_id -> target_id)."""
        for conn in self._adj_up.get(source_id, []):
            if conn.target_id == target_id:
                return conn
        return None

    def _creates_cycle(self, source_id: str, target_id: str) -> bool:
        """Return True if adding source_id -> target_id would introduce a cycle."""
        if source_id == target_id:
            return True
        visited: Set[str] = {target_id}
        queue: deque[str] = deque([target_id])
        while queue:
            curr = queue.popleft()
            for conn in self._adj_up.get(curr, []):
                nxt = conn.target_id
                if nxt == source_id:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def connect(
        self,
        source_id: str,
        target_id: str,
        strength: float = 1.0,
        now: Optional[datetime] = None,
    ) -> Connection:
        """Create or update a directed upward edge from prerequisite to higher application."""
        if source_id not in self.concepts:
            raise KeyError(f"Source concept {source_id!r} not found in graph")
        if target_id not in self.concepts:
            raise KeyError(f"Target concept {target_id!r} not found in graph")
        if source_id == target_id:
            raise ValueError(f"Cannot connect concept {source_id!r} to itself")

        clamped = round(max(0.0, min(1.0, float(strength))), 4)
        now_iso = _now_iso(now)

        existing = self.get_connection(source_id, target_id)
        if existing is not None:
            existing.strength = clamped
            if existing.last_reviewed is not None:
                existing.last_reviewed = now_iso
            else:
                existing.created_at = now_iso
            return existing

        if self._creates_cycle(source_id, target_id):
            raise ValueError(
                f"Connection {source_id!r} -> {target_id!r} would create a cycle in the prerequisite DAG"
            )

        conn = Connection(
            source_id=source_id,
            target_id=target_id,
            strength=clamped,
            created_at=now_iso,
        )
        self.connections.append(conn)
        self._index_connection(conn)
        return conn

    def disconnect(self, source_id: str, target_id: str) -> Connection:
        """Remove a directed connection from the graph."""
        conn = self.get_connection(source_id, target_id)
        if conn is None:
            raise KeyError(f"Connection {source_id!r} -> {target_id!r} does not exist")

        self.connections.remove(conn)
        if conn in self._adj_up.get(source_id, []):
            self._adj_up[source_id].remove(conn)
        if conn in self._adj_down.get(target_id, []):
            self._adj_down[target_id].remove(conn)
        return conn

    def remove_concept(self, concept_id: str) -> ConceptBlueprint:
        """Remove a concept and all incident connections cleanly."""
        if concept_id not in self.concepts:
            raise KeyError(f"Concept {concept_id!r} not found in graph")

        for conn in list(self._adj_up.get(concept_id, [])):
            self.disconnect(conn.source_id, conn.target_id)
        for conn in list(self._adj_down.get(concept_id, [])):
            self.disconnect(conn.source_id, conn.target_id)

        self._adj_up.pop(concept_id, None)
        self._adj_down.pop(concept_id, None)
        return self.concepts.pop(concept_id)

    def _merge_concept_preserving_state(self, incoming: ConceptBlueprint) -> ConceptBlueprint:
        """Insert a new concept or update an existing one without wiping learning metadata."""
        existing = self.concepts.get(incoming.id)
        if existing is None:
            return self.add_concept(incoming)

        existing.title = incoming.title or existing.title
        if incoming.topic and (incoming.topic != "General" or existing.topic == "General"):
            existing.topic = incoming.topic
        if incoming.content and len(incoming.content) >= len(existing.content):
            existing.content = incoming.content
        existing.updated_at = _now_iso()
        return existing

    def add_bundle(
        self,
        concept: ConceptBlueprint,
        prerequisites: List[Union[ConceptBlueprint, str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Crystallize a learning session: register concept + prerequisites and wire upward.

        - Preserves learning metadata (half-life, review history) on existing concepts.
        - Preserves existing edge strengths/history if a prerequisite link already exists.
        - Prerequisites may be full ConceptBlueprints, dicts, or existing concept ID strings.
        """
        main_bp = self._merge_concept_preserving_state(concept)

        saved_prereqs: List[ConceptBlueprint] = []
        wired_connections: List[Connection] = []

        for item in prerequisites:
            if isinstance(item, str):
                pid = item if item in self.concepts else slugify(item)
                if pid not in self.concepts:
                    raise KeyError(f"Referenced prerequisite concept {item!r} not found in graph")
                bp = self.concepts[pid]
            elif isinstance(item, dict):
                pid = item.get("id") or slugify(item["title"])
                if pid in self.concepts and not item.get("content"):
                    bp = self.concepts[pid]
                else:
                    candidate_bp = ConceptBlueprint(
                        id=pid,
                        title=item["title"],
                        topic=item.get("topic") or main_bp.topic,
                        content=item["content"],
                        intrinsic_score=float(item.get("intrinsic_score", 1.0)),
                    )
                    bp = self._merge_concept_preserving_state(candidate_bp)
            else:
                pid = item.id or slugify(item.title)
                item.id = pid
                bp = self._merge_concept_preserving_state(item)

            saved_prereqs.append(bp)

            existing_conn = self.get_connection(bp.id, main_bp.id)
            if existing_conn is not None:
                wired_connections.append(existing_conn)
            else:
                conn = self.connect(source_id=bp.id, target_id=main_bp.id, strength=1.0)
                wired_connections.append(conn)

        return {
            "concept": main_bp,
            "prerequisites": saved_prereqs,
            "connections": wired_connections,
        }

    # -------------------------------------------------------------------------
    # Graph Inspection, Search, Topology & Diagnostics
    # -------------------------------------------------------------------------

    def get_outgoing(self, concept_id: str) -> List[Connection]:
        """Look UP: O(1) return of all upward connections where concept is applied."""
        return list(self._adj_up.get(concept_id, []))

    def get_incoming(self, concept_id: str) -> List[Connection]:
        """Look DOWN: O(1) return of all downward connections feeding this concept."""
        return list(self._adj_down.get(concept_id, []))

    def get_components(self, concept_id: str) -> List[Connection]:
        """Alias for get_incoming (look DOWN)."""
        return self.get_incoming(concept_id)

    def get_applications(self, concept_id: str) -> List[Connection]:
        """Alias for get_outgoing (look UP)."""
        return self.get_outgoing(concept_id)

    def get_foundational_ingredients(self, concept_id: str) -> List[ConceptBlueprint]:
        """Direct prerequisite blueprints needed to build this concept from below."""
        return [
            self.concepts[c.source_id]
            for c in self.get_incoming(concept_id)
            if c.source_id in self.concepts
        ]

    def get_higher_applications(self, concept_id: str) -> List[ConceptBlueprint]:
        """Direct higher-level blueprints that this concept points UP into."""
        return [
            self.concepts[c.target_id]
            for c in self.get_outgoing(concept_id)
            if c.target_id in self.concepts
        ]

    def get_downstream_concepts(self, concept_id: str) -> List[ConceptBlueprint]:
        """All concepts directly or transitively built on top of concept_id (BFS upward)."""
        visited: Set[str] = {concept_id}
        queue: deque[str] = deque([concept_id])
        downstream: List[ConceptBlueprint] = []

        while queue:
            curr = queue.popleft()
            for conn in self._adj_up.get(curr, []):
                nxt = conn.target_id
                if nxt not in visited and nxt in self.concepts:
                    visited.add(nxt)
                    downstream.append(self.concepts[nxt])
                    queue.append(nxt)
        return downstream

    def get_upstream_concepts(self, concept_id: str) -> List[ConceptBlueprint]:
        """All concepts directly or transitively supporting concept_id (BFS downward)."""
        visited: Set[str] = {concept_id}
        queue: deque[str] = deque([concept_id])
        upstream: List[ConceptBlueprint] = []

        while queue:
            curr = queue.popleft()
            for conn in self._adj_down.get(curr, []):
                prv = conn.source_id
                if prv not in visited and prv in self.concepts:
                    visited.add(prv)
                    upstream.append(self.concepts[prv])
                    queue.append(prv)
        return upstream

    def topological_layers(self) -> Dict[str, int]:
        """Compute longest-path prerequisite layer for every concept (0 = pure foundation)."""
        in_degree: Dict[str, int] = {}
        for cid in self.concepts:
            in_degree[cid] = sum(
                1 for conn in self._adj_down.get(cid, []) if conn.source_id in self.concepts
            )

        layers: Dict[str, int] = {}
        queue: deque[str] = deque(sorted(cid for cid, deg in in_degree.items() if deg == 0))
        for cid in queue:
            layers[cid] = 0

        while queue:
            u = queue.popleft()
            for conn in self._adj_up.get(u, []):
                v = conn.target_id
                if v not in self.concepts:
                    continue
                layers[v] = max(layers.get(v, 0), layers[u] + 1)
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        fallback_layer = (max(layers.values()) + 1) if layers else 0
        for cid in self.concepts:
            if cid not in layers:
                layers[cid] = fallback_layer
        return layers

    def diagnose_concept(
        self,
        concept_id: str,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Build a ranked downward fault-isolation cone for a concept.

        Lists every supporting joint reachable below concept_id, ordered from weakest
        live strength first so a Socratic tutor can pinpoint the exact broken link.
        """
        concept = self.concepts.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept {concept_id!r} not found")

        ref_now = _resolve_now(now)
        visited_nodes: Set[str] = {concept_id}
        visited_edges: Set[Tuple[str, str]] = set()
        queue: deque[Tuple[str, int]] = deque([(concept_id, 1)])
        cone_joints: List[Dict[str, Any]] = []

        while queue:
            curr, depth = queue.popleft()
            for conn in self._adj_down.get(curr, []):
                edge_key = (conn.source_id, conn.target_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    eff = self.effective_connection_strength(conn, now=ref_now)
                    cone_joints.append({
                        "source_id": conn.source_id,
                        "source_title": self.concepts[conn.source_id].title if conn.source_id in self.concepts else conn.source_id,
                        "target_id": conn.target_id,
                        "target_title": self.concepts[conn.target_id].title if conn.target_id in self.concepts else conn.target_id,
                        "hops_below": depth,
                        "recorded_strength": conn.strength,
                        "effective_strength": eff,
                        "fault": round(1.0 - conn.strength, 4),
                        "decay": round(max(0.0, conn.strength - eff), 4),
                        "half_life_days": conn.half_life,
                        "review_count": conn.review_count,
                    })
                if conn.source_id not in visited_nodes and conn.source_id in self.concepts:
                    visited_nodes.add(conn.source_id)
                    queue.append((conn.source_id, depth + 1))

        cone_joints.sort(key=lambda j: (j["effective_strength"], j["hops_below"], j["source_id"]))

        direct_prereqs = [
            {
                "id": ing.id,
                "title": ing.title,
                "topic": ing.topic,
                "effective_strength": self.effective_connection_strength(
                    self.get_connection(ing.id, concept_id), now=ref_now
                ),
            }
            for ing in self.get_foundational_ingredients(concept_id)
            if self.get_connection(ing.id, concept_id) is not None
        ]
        direct_prereqs.sort(key=lambda p: p["effective_strength"])

        return {
            "concept_id": concept.id,
            "title": concept.title,
            "topic": concept.topic,
            "stability": self.calculate_concept_stability(concept_id, now=ref_now),
            "direct_prerequisites": direct_prereqs,
            "supporting_cone": cone_joints,
            "weakest_joint": cone_joints[0] if cone_joints else None,
        }

    def find_weakest_supporting_joint(
        self,
        concept_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[Tuple[Connection, float, int]]:
        """Locate the single weakest effective connection anywhere in concept_id's downward cone."""
        diag = self.diagnose_concept(concept_id, now=now)
        weakest = diag["weakest_joint"]
        if weakest is None:
            return None
        conn = self.get_connection(weakest["source_id"], weakest["target_id"])
        if conn is None:
            return None
        return (conn, weakest["effective_strength"], weakest["hops_below"])

    def search_concepts(
        self,
        query: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[ConceptBlueprint]:
        """Filter concepts by topic and/or case-insensitive text search."""
        results = list(self.concepts.values())
        if topic:
            t_lower = topic.lower()
            results = [c for c in results if t_lower in c.topic.lower()]
        if query:
            q_lower = query.lower()
            results = [
                c
                for c in results
                if q_lower in c.id.lower()
                or q_lower in c.title.lower()
                or q_lower in c.topic.lower()
                or q_lower in c.content.lower()
            ]
        return sorted(results, key=lambda c: c.id)

    def summary(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Compute a bird's-eye cognitive health snapshot of the entire graph."""
        ref_now = _resolve_now(now)
        layers = self.topological_layers()
        max_layer = max(layers.values()) if layers else -1

        frontier_leaves = []
        isolated_concepts = []
        keystones = []
        live_scores: List[float] = []
        active_faults: List[Dict[str, Any]] = []
        most_decayed: List[Dict[str, Any]] = []

        for cid, c in sorted(self.concepts.items()):
            out_edges = self.get_outgoing(cid)
            in_edges = self.get_incoming(cid)
            load = len(self.get_downstream_concepts(cid))

            if load > 0:
                keystones.append({
                    "id": cid,
                    "title": c.title,
                    "structural_load": load,
                    "stability": self.calculate_concept_stability(cid, now=ref_now)["stability"],
                })

            if not out_edges and not in_edges:
                isolated_concepts.append({"id": cid, "title": c.title, "topic": c.topic})

            if not out_edges:
                eff = self.effective_concept_intrinsic(c, now=ref_now)
                live_scores.append(eff)
                frontier_leaves.append({
                    "id": cid,
                    "title": c.title,
                    "recorded_strength": c.intrinsic_score,
                    "effective_strength": eff,
                    "half_life_days": c.half_life,
                })
                if c.intrinsic_score < 1.0:
                    active_faults.append({
                        "type": "concept",
                        "concept_id": cid,
                        "recorded_strength": c.intrinsic_score,
                        "effective_strength": eff,
                    })
                decay = round(max(0.0, c.intrinsic_score - eff), 4)
                if decay > 0.01:
                    most_decayed.append({
                        "type": "concept",
                        "concept_id": cid,
                        "decay": decay,
                        "effective_strength": eff,
                    })

        for conn in self.connections:
            eff = self.effective_connection_strength(conn, now=ref_now)
            live_scores.append(eff)
            if conn.strength < 1.0:
                active_faults.append({
                    "type": "joint",
                    "source_id": conn.source_id,
                    "target_id": conn.target_id,
                    "recorded_strength": conn.strength,
                    "effective_strength": eff,
                })
            decay = round(max(0.0, conn.strength - eff), 4)
            if decay > 0.01:
                most_decayed.append({
                    "type": "joint",
                    "source_id": conn.source_id,
                    "target_id": conn.target_id,
                    "decay": decay,
                    "effective_strength": eff,
                })

        active_faults.sort(key=lambda x: (x["recorded_strength"], x["effective_strength"]))
        most_decayed.sort(key=lambda x: x["decay"], reverse=True)
        keystones.sort(key=lambda x: (-x["structural_load"], x["stability"]))

        integrity = round(sum(live_scores) / len(live_scores), 4) if live_scores else 1.0

        return {
            "concept_count": len(self.concepts),
            "connection_count": len(self.connections),
            "layer_count": max_layer + 1 if self.concepts else 0,
            "network_integrity": integrity,
            "frontier_leaves": frontier_leaves,
            "keystones": keystones[:5],
            "active_faults": active_faults[:5],
            "most_decayed": most_decayed[:5],
            "isolated_concepts": isolated_concepts,
        }

    # -------------------------------------------------------------------------
    # Continuous Memory Decay, Stability & Dual-Urgency Study Selection
    # -------------------------------------------------------------------------

    def connection_days_elapsed(self, conn: Connection, now: Optional[datetime] = None) -> float:
        ref_now = _resolve_now(now)
        ref_ts = _parse_iso(conn.last_reviewed or conn.created_at)
        return max(0.0, (ref_now - ref_ts).total_seconds() / 86400.0)

    def _effective_connection_half_life(self, conn: Connection) -> float:
        """Breadth-anchored half-life: concepts supporting more contexts decay more slowly."""
        k = len(self._adj_up.get(conn.source_id, []))
        return conn.half_life * breadth_multiplier(k)

    def _connection_retrievability(self, conn: Connection, now: Optional[datetime] = None) -> float:
        elapsed = self.connection_days_elapsed(conn, now=now)
        return decay_factor(elapsed, self._effective_connection_half_life(conn))

    def effective_connection_strength(self, conn: Connection, now: Optional[datetime] = None) -> float:
        """Live time-decayed strength of a connection in [0.0, 1.0]."""
        return round(conn.strength * self._connection_retrievability(conn, now=now), 4)

    def concept_days_elapsed(self, concept: ConceptBlueprint, now: Optional[datetime] = None) -> float:
        ref_now = _resolve_now(now)
        ref_ts = _parse_iso(concept.last_reviewed or concept.created_at)
        return max(0.0, (ref_now - ref_ts).total_seconds() / 86400.0)

    def _concept_retrievability(self, concept: ConceptBlueprint, now: Optional[datetime] = None) -> float:
        elapsed = self.concept_days_elapsed(concept, now=now)
        return decay_factor(elapsed, concept.half_life)

    def effective_concept_intrinsic(self, concept: ConceptBlueprint, now: Optional[datetime] = None) -> float:
        """Live time-decayed intrinsic mastery of a frontier leaf concept."""
        return round(concept.intrinsic_score * self._concept_retrievability(concept, now=now), 4)

    def calculate_concept_stability(
        self,
        concept_id: str,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Live concept stability = (Average Live Outgoing Strength) * (1 + 0.5 * ln(1 + K)).

        Frontier leaves (K = 0) fall back to their live intrinsic score.
        """
        concept = self.concepts.get(concept_id)
        if not concept:
            raise KeyError(f"Concept {concept_id!r} not found")

        ref_now = _resolve_now(now)
        outgoing = self.get_outgoing(concept_id)
        k = len(outgoing)

        if k == 0:
            eff_intrinsic = self.effective_concept_intrinsic(concept, now=ref_now)
            return {
                "concept_id": concept_id,
                "breadth_k": 0,
                "avg_strength": eff_intrinsic,
                "min_strength": eff_intrinsic,
                "breadth_bonus": 1.0,
                "stability": eff_intrinsic,
                "is_frontier_leaf": True,
            }

        eff_strengths = [self.effective_connection_strength(c, now=ref_now) for c in outgoing]
        avg_s = sum(eff_strengths) / k
        min_s = min(eff_strengths)
        bonus = breadth_multiplier(k)
        stability = avg_s * bonus

        return {
            "concept_id": concept_id,
            "breadth_k": k,
            "avg_strength": round(avg_s, 4),
            "min_strength": round(min_s, 4),
            "breadth_bonus": round(bonus, 4),
            "stability": round(stability, 4),
            "is_frontier_leaf": False,
        }

    def _upstream_decay_cone(
        self,
        start_node: str,
        now: datetime,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Compute total damped decay mass recoverable by implicit confirmation below start_node.

        Exact mathematical dual of _propagate_implicit_confirmation().
        """
        total_leverage = 0.0
        refresh_targets: List[Dict[str, Any]] = []
        visited_edges: Set[Tuple[str, str]] = set()
        queued_nodes: Set[str] = {start_node}
        queue: deque[Tuple[str, int]] = deque([(start_node, 1)])

        while queue:
            curr, depth = queue.popleft()
            credit = IMPLICIT_DAMPING ** depth
            if credit < MIN_IMPLICIT_CREDIT:
                continue

            for conn in self._adj_down.get(curr, []):
                edge_key = (conn.source_id, conn.target_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    eff_w = self.effective_connection_strength(conn, now=now)
                    edge_decay = max(0.0, conn.strength - eff_w)
                    if edge_decay > 1e-4:
                        total_leverage += credit * edge_decay
                        refresh_targets.append({
                            "source_id": conn.source_id,
                            "target_id": conn.target_id,
                            "depth": depth,
                            "effective_strength": eff_w,
                            "decay": round(edge_decay, 4),
                        })

                if conn.source_id not in queued_nodes and conn.source_id in self.concepts:
                    queued_nodes.add(conn.source_id)
                    queue.append((conn.source_id, depth + 1))

        return total_leverage, refresh_targets

    def get_study_candidates(
        self,
        top_k: int = 3,
        now: Optional[datetime] = None,
    ) -> List[StudyCandidate]:
        """Surface the highest-priority joints and frontier leaves across the entire graph.

        Unified dual urgency (zero thresholds):
            Urgency(x) = Fault_Repair(x) + Implicit_Confirmation_Leverage(x)
            - Fault_Repair(x) = (1 - w(x)) * B(downstream_load(x))
            - Leverage(x)     = decay(x) + sum_{e in upstream(x)} 0.5^d * decay(e)

        Known faults scale UPWARD (surgical bottom-up repair of load-bearing links).
        Pure time decay sums DOWNWARD (efficient top-down checks that refresh whole cones).
        """
        ref_now = _resolve_now(now)
        ranked: List[Tuple[Tuple[float, float, int, float], StudyCandidate]] = []

        # 1. Every directed upward joint (u -> v)
        for conn in self.connections:
            source = self.concepts.get(conn.source_id)
            target = self.concepts.get(conn.target_id)
            if source is None or target is None:
                continue

            eff_w = self.effective_connection_strength(conn, now=ref_now)
            own_fault = max(0.0, 1.0 - conn.strength)
            own_decay = max(0.0, conn.strength - eff_w)

            downstream = [target] + self.get_downstream_concepts(target.id)
            load = len(downstream)
            fault_score = own_fault * breadth_multiplier(load)

            upstream_lev, refresh_targets = self._upstream_decay_cone(source.id, ref_now)
            decay_leverage = own_decay + upstream_lev
            urgency = fault_score + decay_leverage

            if urgency <= 1e-6:
                reason = "fresh"
            elif fault_score >= decay_leverage:
                reason = "repair"
            else:
                reason = "leverage"

            metrics = self.calculate_concept_stability(source.id, now=ref_now)
            target_prereqs = [
                {
                    "id": inc.source_id,
                    "title": self.concepts[inc.source_id].title,
                    "recorded_strength": inc.strength,
                    "effective_strength": self.effective_connection_strength(inc, now=ref_now),
                }
                for inc in self.get_incoming(target.id)
                if inc.source_id in self.concepts
            ]

            candidate = StudyCandidate(
                concept=source,
                urgency_score=round(urgency, 4),
                fault_score=round(fault_score, 4),
                decay_leverage=round(decay_leverage, 4),
                review_reason=reason,
                stability_score=metrics["stability"],
                effective_strength=eff_w,
                structural_load=load,
                frayed_target=target,
                frayed_connection=conn,
                downstream_impact=downstream,
                foundational_ingredients=self.get_foundational_ingredients(source.id),
                target_prerequisites=target_prereqs,
                implicit_refresh_targets=refresh_targets,
            )

            ref_ts = conn.last_reviewed or conn.created_at
            age_seconds = max(0.0, (ref_now - _parse_iso(ref_ts)).total_seconds())
            ranked.append(((candidate.urgency_score, -candidate.stability_score, load, age_seconds), candidate))

        # 2. Every frontier leaf concept (K = 0 outgoing edges)
        for cid, concept in self.concepts.items():
            if self._adj_up.get(cid):
                continue

            eff_w = self.effective_concept_intrinsic(concept, now=ref_now)
            own_fault = max(0.0, 1.0 - concept.intrinsic_score)
            own_decay = max(0.0, concept.intrinsic_score - eff_w)

            fault_score = own_fault
            upstream_lev, refresh_targets = self._upstream_decay_cone(cid, ref_now)
            decay_leverage = own_decay + upstream_lev
            urgency = fault_score + decay_leverage

            if urgency <= 1e-6:
                reason = "fresh"
            elif fault_score >= decay_leverage:
                reason = "repair"
            else:
                reason = "leverage"

            metrics = self.calculate_concept_stability(cid, now=ref_now)
            candidate = StudyCandidate(
                concept=concept,
                urgency_score=round(urgency, 4),
                fault_score=round(fault_score, 4),
                decay_leverage=round(decay_leverage, 4),
                review_reason=reason,
                stability_score=metrics["stability"],
                effective_strength=eff_w,
                structural_load=0,
                frayed_target=None,
                frayed_connection=None,
                downstream_impact=[],
                foundational_ingredients=self.get_foundational_ingredients(cid),
                target_prerequisites=[],
                implicit_refresh_targets=refresh_targets,
            )

            ref_ts = concept.last_reviewed or concept.created_at
            age_seconds = max(0.0, (ref_now - _parse_iso(ref_ts)).total_seconds())
            ranked.append(((candidate.urgency_score, -candidate.stability_score, 0, age_seconds), candidate))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [cand for _, cand in ranked[:top_k]]

    # -------------------------------------------------------------------------
    # Socratic Review & Asymmetric Fractional Implicit Confirmation
    # -------------------------------------------------------------------------

    def _propagate_implicit_confirmation(
        self,
        start_node: str,
        score: float,
        now: datetime,
    ) -> List[ImplicitReinforcement]:
        """Asymmetric downward propagation (Fractional Implicit Repetition).

        Strong mastery of a higher node sends damped credit (0.5^depth) to supporting
        prerequisite connections ONLY when score exceeds an edge's live strength.
        Low scores NEVER propagate downward—preserving strict fault isolation.
        """
        now_iso = _now_iso(now)
        reinforced: List[ImplicitReinforcement] = []
        visited_edges: Set[Tuple[str, str]] = set()
        queued_nodes: Set[str] = {start_node}
        queue: deque[Tuple[str, int]] = deque([(start_node, 1)])

        while queue:
            curr, depth = queue.popleft()
            credit = IMPLICIT_DAMPING ** depth
            if credit < MIN_IMPLICIT_CREDIT:
                continue

            for conn in self._adj_down.get(curr, []):
                edge_key = (conn.source_id, conn.target_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    eff_w = self.effective_connection_strength(conn, now=now)
                    if score > eff_w:
                        r = self._connection_retrievability(conn, now=now)
                        new_w = round(eff_w + credit * (score - eff_w), 4)
                        new_hl = (
                            update_half_life(conn.half_life, score, r, weight=credit)
                            if score > 0.5
                            else conn.half_life
                        )
                        conn.strength = new_w
                        conn.half_life = new_hl
                        conn.last_reviewed = now_iso
                        reinforced.append(
                            ImplicitReinforcement(
                                source_id=conn.source_id,
                                target_id=conn.target_id,
                                depth=depth,
                                previous_effective_strength=eff_w,
                                new_strength=new_w,
                                new_half_life=new_hl,
                            )
                        )

                if conn.source_id not in queued_nodes and conn.source_id in self.concepts:
                    queued_nodes.add(conn.source_id)
                    queue.append((conn.source_id, depth + 1))

        return reinforced

    def review_joint(
        self,
        source_id: str,
        target_id: str,
        score: float,
        now: Optional[datetime] = None,
        propagate: bool = True,
    ) -> ReviewResult:
        """Record a Socratic review for a specific directed joint (prerequisite -> target)."""
        conn = self.get_connection(source_id, target_id)
        if conn is None:
            raise KeyError(f"Connection {source_id!r} -> {target_id!r} does not exist")

        ref_now = _resolve_now(now)
        now_iso = _now_iso(ref_now)
        clamped = round(max(0.0, min(1.0, float(score))), 4)

        prev_eff = self.effective_connection_strength(conn, now=ref_now)
        retrievability = self._connection_retrievability(conn, now=ref_now)

        conn.half_life = update_half_life(conn.half_life, clamped, retrievability, weight=1.0)
        conn.strength = clamped
        conn.last_reviewed = now_iso
        conn.review_count += 1

        reinforced = (
            self._propagate_implicit_confirmation(source_id, clamped, now=ref_now)
            if propagate
            else []
        )
        stability = self.calculate_concept_stability(source_id, now=ref_now)

        return ReviewResult(
            mode="joint",
            concept_id=source_id,
            source_id=source_id,
            target_id=target_id,
            previous_effective_strength=prev_eff,
            new_strength=conn.strength,
            new_half_life=conn.half_life,
            review_count=conn.review_count,
            last_reviewed=now_iso,
            stability=stability,
            implicitly_reinforced=reinforced,
        )

    def review_concept(
        self,
        concept_id: str,
        score: float,
        now: Optional[datetime] = None,
        propagate: bool = True,
    ) -> ReviewResult:
        """Record a direct Socratic review for a concept (primarily frontier leaves)."""
        concept = self.concepts.get(concept_id)
        if concept is None:
            raise KeyError(f"Concept {concept_id!r} not found")

        ref_now = _resolve_now(now)
        now_iso = _now_iso(ref_now)
        clamped = round(max(0.0, min(1.0, float(score))), 4)

        prev_eff = self.effective_concept_intrinsic(concept, now=ref_now)
        retrievability = self._concept_retrievability(concept, now=ref_now)

        concept.half_life = update_half_life(concept.half_life, clamped, retrievability, weight=1.0)
        concept.intrinsic_score = clamped
        concept.last_reviewed = now_iso
        concept.review_count += 1

        reinforced = (
            self._propagate_implicit_confirmation(concept_id, clamped, now=ref_now)
            if propagate
            else []
        )
        stability = self.calculate_concept_stability(concept_id, now=ref_now)

        return ReviewResult(
            mode="concept",
            concept_id=concept_id,
            previous_effective_strength=prev_eff,
            new_strength=concept.intrinsic_score,
            new_half_life=concept.half_life,
            review_count=concept.review_count,
            last_reviewed=now_iso,
            stability=stability,
            implicitly_reinforced=reinforced,
        )

    def review_bundle(
        self,
        target_id: str,
        component_scores: Dict[str, float],
        now: Optional[datetime] = None,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        """Apply asymmetric fault-isolation scores across inbound joints feeding target_id."""
        if target_id not in self.concepts:
            raise KeyError(f"Target concept {target_id!r} not found")

        ref_now = _resolve_now(now)
        results: List[ReviewResult] = []
        skipped: List[str] = []

        for src_id, score in component_scores.items():
            if self.get_connection(src_id, target_id) is None:
                skipped.append(src_id)
                continue
            results.append(
                self.review_joint(src_id, target_id, score=score, now=ref_now, propagate=propagate)
            )

        return {
            "target_id": target_id,
            "target_title": self.concepts[target_id].title,
            "updated_joints": len(results),
            "skipped_sources": skipped,
            "joints": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "previous_effective_strength": r.previous_effective_strength,
                    "new_strength": r.new_strength,
                    "new_half_life": r.new_half_life,
                    "implicitly_reinforced": [ir.model_dump() for ir in r.implicitly_reinforced],
                }
                for r in results
            ],
        }