"""Tests for Socratic tutor prompt generation and scaffolding."""
from nexus.engine import NexusEngine
from nexus.models import ConceptBlueprint
from nexus.tutor import SocraticTutor


def test_socratic_study_brief_and_probe():
    engine = NexusEngine()
    vae = ConceptBlueprint(
        id="vae",
        title="Variational Autoencoder",
        topic="Deep Learning",
        content="# Variational Autoencoders\n\nELBO optimization with reparameterization trick.",
    )
    bayes = ConceptBlueprint(
        id="bayes",
        title="Bayes' Theorem",
        topic="Probability",
        content="# Bayes' Theorem\n\nPosterior proportional to likelihood times prior.",
    )
    engine.add_concept(vae)
    engine.add_concept(bayes)
    engine.connect(
        source_id="bayes",
        target_id="vae",
        strength=0.45,
    )

    candidates = engine.get_study_candidates(top_k=1)
    assert len(candidates) == 1
    candidate = candidates[0]

    brief = SocraticTutor.build_study_brief(candidate)
    assert "Socratic Study Target: Bayes' Theorem" in brief
    assert "Weak Upward Connection" in brief
    assert "Variational Autoencoder" in brief
    assert "0.45 / 1.00" in brief
    assert "STRICTLY NO analogies or metaphors" in brief

    probe = SocraticTutor.generate_joint_probe(candidate)
    assert "Variational Autoencoder" in probe
    assert "Without using analogies" in probe
    assert "0.45" in probe
