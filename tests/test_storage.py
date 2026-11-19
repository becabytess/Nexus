"""Tests for Nexus atomic graph persistence, legacy loading, and Markdown export."""
import json
import os
import tempfile

from nexus.engine import NexusEngine
from nexus.models import ConceptBlueprint
from nexus.storage import export_markdown, load_graph, save_graph


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_nexus.json")

        engine = NexusEngine()
        bp1 = ConceptBlueprint(
            id="information-entropy",
            title="Information Entropy",
            topic="Information Theory",
            content=r"# Information Entropy\n\n$H(X) = -\sum p(x) \log p(x)$.",
            intrinsic_score=0.9,
            half_life=4.5,
        )
        bp2 = ConceptBlueprint(
            id="cross-entropy-loss",
            title="Cross Entropy Loss",
            topic="Machine Learning",
            content="# Cross Entropy Loss\n\nMeasures divergence between distributions.",
        )
        engine.add_concept(bp1)
        engine.add_concept(bp2)
        engine.connect("information-entropy", "cross-entropy-loss", strength=0.88)
        engine.review_joint("information-entropy", "cross-entropy-loss", score=0.92)

        save_graph(engine, file_path)
        assert os.path.exists(file_path)

        loaded = load_graph(file_path)
        assert len(loaded.concepts) == 2
        assert loaded.concepts["information-entropy"].half_life == 4.5
        assert len(loaded.connections) == 1
        conn = loaded.get_connection("information-entropy", "cross-entropy-loss")
        assert conn is not None
        assert conn.strength == 0.92
        assert conn.review_count == 1


def test_load_legacy_v1_schema_backward_compatible():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "legacy.json")
        legacy_payload = {
            "version": "1.0",
            "concepts": {
                "a": {"id": "a", "title": "A", "topic": "Math", "content": "# A"},
                "b": {"id": "b", "title": "B", "topic": "Math", "content": "# B"},
            },
            "connections": [
                {"source_id": "a", "target_id": "b", "strength": 0.65, "review_count": 2}
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f)

        loaded = load_graph(file_path)
        assert len(loaded.concepts) == 2
        conn = loaded.get_connection("a", "b")
        assert conn is not None
        assert conn.strength == 0.65
        assert conn.half_life == 1.0


def test_export_markdown_wiki():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = NexusEngine()
        engine.add_concept(
            ConceptBlueprint(
                id="bayes",
                title="Bayes' Theorem",
                topic="Probability",
                content=r"# Bayes\n\n$$P(z|x) = \frac{P(x|z)P(z)}{P(x)}$$",
            )
        )
        engine.add_concept(
            ConceptBlueprint(id="vae", title="VAEs", topic="Deep Learning", content="# VAEs\n\nELBO.")
        )
        engine.connect("bayes", "vae", strength=0.75)

        out_dir = os.path.join(tmpdir, "wiki")
        res = export_markdown(engine, out_dir)
        assert res["files_written"] == 3
        assert os.path.exists(os.path.join(out_dir, "INDEX.md"))
        assert os.path.exists(os.path.join(out_dir, "bayes.md"))
        assert os.path.exists(os.path.join(out_dir, "vae.md"))

        with open(os.path.join(out_dir, "bayes.md"), "r", encoding="utf-8") as f:
            bayes_md = f.read()
        assert r"\frac{P(x|z)P(z)}{P(x)}" in bayes_md
        assert "[VAEs](vae.md)" in bayes_md

        with open(os.path.join(out_dir, "INDEX.md"), "r", encoding="utf-8") as f:
            index_md = f.read()
        assert "Layer 0 — Foundations" in index_md
        assert "[Bayes' Theorem](bayes.md)" in index_md


def test_load_nonexistent_returns_empty_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "missing.json")
        engine = load_graph(non_existent)
        assert isinstance(engine, NexusEngine)
        assert len(engine.concepts) == 0
        assert len(engine.connections) == 0