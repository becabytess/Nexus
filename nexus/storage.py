"""Persistence module for Nexus graphs using clean human-readable JSON."""
from __future__ import annotations

import json
import os
from typing import Optional

from nexus.engine import NexusEngine
from nexus.models import ConceptBlueprint, Connection, _now_iso

DEFAULT_GRAPH_FILE = "nexus_graph.json"


def save_graph(engine: NexusEngine, path: str = DEFAULT_GRAPH_FILE) -> None:
    """Save the Nexus concept blueprints and connection graph to JSON."""
    data = {
        "version": "1.0",
        "updated_at": _now_iso(),
        "concepts": {cid: c.model_dump() for cid, c in engine.concepts.items()},
        "connections": [conn.model_dump() for conn in engine.connections],
    }

    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_graph(path: str = DEFAULT_GRAPH_FILE) -> NexusEngine:
    """Load a Nexus graph from a JSON file, or return a fresh engine if file does not exist."""
    engine = NexusEngine()
    if not os.path.exists(path):
        return engine

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concepts_data = data.get("concepts", {})
    for cid, c_dict in concepts_data.items():
        engine.concepts[cid] = ConceptBlueprint(**c_dict)

    connections_data = data.get("connections", [])
    for conn_dict in connections_data:
        engine.connections.append(Connection(**conn_dict))

    return engine
