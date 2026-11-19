"""Tests for Socratic tutor briefs, diagnostic cones, probes, and review summaries."""
from nexus.engine import NexusEngine
from nexus.models import ConceptBlueprint
from nexus.tutor import SocraticTutor


def test_socratic_study_brief_includes_full_target_and_siblings():
    engine = NexusEngine()
    long_target_body = (
        "# Variational Autoencoders\n\n"
        + ("Introductory density context. " * 25)
        + "\n\n## ELBO Derivation\nUNIQUE_BOTTOM_MARKER_ELBO"
    )
    vae = ConceptBlueprint(id="vae", title="Variational Autoencoder", topic="Deep Learning", content=long_target_body)
    bayes = ConceptBlueprint(id="bayes", title="Bayes' Theorem", topic="Probability", content="# Bayes\n\nPosterior.")
    kl = ConceptBlueprint(id="kl", title="KL Divergence", topic="Info Theory", content="# KL\n\nRelative entropy.")

    engine.add_bundle(vae, [bayes, kl])
    engine.connect("bayes", "vae", strength=0.45)

    candidate = engine.get_study_candidates(top_k=1)[0]
    brief = SocraticTutor.build_study_brief(candidate)

    assert "Socratic Study Target: Bayes' Theorem" in brief
    assert "Surgical Bottom-Up Repair" in brief
    assert "UNIQUE_BOTTOM_MARKER_ELBO" in brief
    assert "KL Divergence" in brief
    assert "STRICTLY NO analogies or metaphors" in brief

    probe = SocraticTutor.generate_joint_probe(candidate)
    assert "Variational Autoencoder" in probe
    assert "Without using analogies" in probe


def test_diagnostic_brief_frontier_leaf_and_review_summary():
    engine = NexusEngine()
    p = ConceptBlueprint(id="p", title="Prereq", content="# Prereq")
    leaf = ConceptBlueprint(id="leaf", title="Frontier Leaf", content="# Frontier", intrinsic_score=0.5)
    engine.add_bundle(leaf, [p])
    engine.connect("p", "leaf", strength=0.6)

    diag = engine.diagnose_concept("leaf")
    diag_brief = SocraticTutor.build_diagnostic_brief(diag)
    assert "Fault-Isolation Cone: Frontier Leaf" in diag_brief
    assert "`p -> leaf`" in diag_brief

    leaf_cand = [
        c for c in engine.get_study_candidates(top_k=5)
        if c.concept.id == "leaf" and c.frayed_connection is None
    ][0]
    brief = SocraticTutor.build_study_brief(leaf_cand)
    assert "Frontier Leaf Evaluation" in brief
    assert "Prereq" in brief

    probe = SocraticTutor.generate_joint_probe(leaf_cand)
    assert "Frontier Leaf" in probe
    assert "Prereq" in probe

    result = engine.review_concept("leaf", score=1.0)
    summary = SocraticTutor.format_review_summary(result)
    assert "Updated frontier concept `leaf`" in summary
    assert "Implicitly reinforced foundations" in summary
    assert "`p -> leaf`" in summary