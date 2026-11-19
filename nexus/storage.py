"""Atomic JSON persistence and standalone Markdown wiki export for Nexus."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

from nexus.engine import NexusEngine
from nexus.models import ConceptBlueprint, Connection, _now_iso

DEFAULT_GRAPH_FILE = "nexus_graph.json"


def save_graph(engine: NexusEngine, path: str = DEFAULT_GRAPH_FILE) -> None:
    """Atomically save concept blueprints and connections to JSON."""
    data = {
        "version": "1.1",
        "updated_at": _now_iso(),
        "concepts": {cid: c.model_dump() for cid, c in engine.concepts.items()},
        "connections": [conn.model_dump() for conn in engine.connections],
    }

    abs_path = os.path.abspath(path)
    dir_name = os.path.dirname(abs_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".nexus_tmp_", suffix=".json", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, abs_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_graph(path: str = DEFAULT_GRAPH_FILE) -> NexusEngine:
    """Load a Nexus graph from JSON, or return an empty engine if the file does not exist."""
    engine = NexusEngine()
    if not os.path.exists(path):
        return engine

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cid, c_dict in data.get("concepts", {}).items():
        engine.concepts[cid] = ConceptBlueprint(**c_dict)

    for conn_dict in data.get("connections", []):
        engine.connections.append(Connection(**conn_dict))

    engine.reindex()
    return engine


def export_markdown(engine: NexusEngine, output_dir: str) -> Dict[str, Any]:
    """Export every concept as a linked standalone Markdown document plus a topological INDEX.md."""
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)

    layers = engine.topological_layers()
    max_layer = max(layers.values()) if layers else -1
    files_written = 0

    for cid, concept in sorted(engine.concepts.items()):
        outgoing = engine.get_outgoing(cid)
        incoming = engine.get_incoming(cid)
        metrics = engine.calculate_concept_stability(cid)

        lines = [concept.content.rstrip(), "", "---", "", "## Nexus Graph Links"]
        lines.append(
            f"- **Topic**: {concept.topic} | **Layer**: {layers.get(cid, 0)} | "
            f"**Stability**: {metrics['stability']:.2f}"
        )

        if incoming:
            lines.append("")
            lines.append("### Built From Below (Prerequisites)")
            for conn in incoming:
                src = engine.concepts.get(conn.source_id)
                if src is None:
                    continue
                eff = engine.effective_connection_strength(conn)
                lines.append(
                    f"- [{src.title}]({src.id}.md) — strength {eff:.2f} (half-life {conn.half_life:.1f}d)"
                )

        if outgoing:
            lines.append("")
            lines.append("### Points Up Into (Higher Applications)")
            for conn in outgoing:
                tgt = engine.concepts.get(conn.target_id)
                if tgt is None:
                    continue
                eff = engine.effective_connection_strength(conn)
                lines.append(
                    f"- [{tgt.title}]({tgt.id}.md) — strength {eff:.2f} (half-life {conn.half_life:.1f}d)"
                )

        lines.append("")
        doc_path = os.path.join(abs_dir, f"{cid}.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        files_written += 1

    index_lines = [
        "# Nexus Cognitive Blueprint Index",
        "",
        f"Concepts: **{len(engine.concepts)}** | Connections: **{len(engine.connections)}**",
        "",
    ]
    by_layer: Dict[int, list[str]] = {}
    for cid, layer in layers.items():
        by_layer.setdefault(layer, []).append(cid)

    for layer_idx in range(max_layer + 1):
        cids = sorted(by_layer.get(layer_idx, []))
        if not cids:
            continue
        if layer_idx == 0:
            heading = "## Layer 0 — Foundations"
        elif layer_idx == max_layer:
            heading = f"## Layer {layer_idx} — Frontier"
        else:
            heading = f"## Layer {layer_idx}"
        index_lines.append(heading)
        index_lines.append("")
        for cid in cids:
            c = engine.concepts[cid]
            m = engine.calculate_concept_stability(cid)
            index_lines.append(
                f"- [{c.title}]({c.id}.md) ({c.topic}) — Stability `{m['stability']:.2f}`, Outgoing `{m['breadth_k']}`"
            )
        index_lines.append("")

    index_path = os.path.join(abs_dir, "INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    files_written += 1

    return {
        "output_dir": abs_dir,
        "files_written": files_written,
        "index_file": index_path,
    }