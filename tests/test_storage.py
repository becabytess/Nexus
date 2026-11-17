"""Tests for Nexus graph persistence."""
import os
import tempfile
from nexus.engine import NexusEngine
from nexus.models import ConceptBlueprint
from nexus.storage import load_graph, save_graph


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_nexus.json")

        engine = NexusEngine()
        bp1 = ConceptBlueprint(
            id="information-entropy",
            title="Information Entropy",
            topic="Information Theory",
            content="# Information Entropy\n\nExpected surprise of a probability distribution: $H(X) = -\\sum p(x) \\log p(x)$.",
            intrinsic_score=1.0,
        )
        bp2 = ConceptBlueprint(
            id="cross-entropy-loss",
            title="Cross Entropy Loss",
            topic="Machine Learning",
            content="# Cross Entropy Loss\n\nMeasures divergence between true and predicted distributions.",
        )
        engine.add_concept(bp1)
        engine.add_concept(bp2)
        engine.connect("information-entropy", "cross-entropy-loss", strength=0.88)

        save_graph(engine, file_path)
        assert os.path.exists(file_path)

        loaded = load_graph(file_path)
        assert len(loaded.concepts) == 2
        assert "information-entropy" in loaded.concepts
        assert "Expected surprise" in loaded.concepts["information-entropy"].content
        assert len(loaded.connections) == 1
        assert loaded.connections[0].strength == 0.88


def test_load_nonexistent_returns_empty_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "missing.json")
        engine = load_graph(non_existent)
        assert isinstance(engine, NexusEngine)
        assert len(engine.concepts) == 0
        assert len(engine.connections) == 0
