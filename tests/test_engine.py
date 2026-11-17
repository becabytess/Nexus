"""Tests for Nexus graph engine: directed upward dependencies, stability, and retrieval."""
import math
import pytest
from nexus.engine import NexusEngine, slugify
from nexus.models import ConceptBlueprint, Connection


def test_slugify():
    assert slugify("Variational Autoencoders (VAEs)") == "variational-autoencoders-vaes"
    assert slugify("Bayes' Theorem") == "bayes-theorem"
    assert slugify("  Linear   Algebra  ") == "linear-algebra"


def test_add_concept_and_lookup():
    engine = NexusEngine()
    bp = ConceptBlueprint(
        id="linear-regression",
        title="Linear Regression",
        topic="Statistics",
        mental_model="Fitting hyperplanes via ordinary least squares minimizing sum of squared residuals.",
        anchor_example="Normal equation (X^T X)^{-1} X^T y",
        pitfalls="Assuming correlation equals causation or ignoring multicollinearity.",
        intrinsic_score=0.9,
    )
    engine.add_concept(bp)

    assert "linear-regression" in engine.concepts
    retrieved = engine.concepts["linear-regression"]
    assert retrieved.title == "Linear Regression"
    assert retrieved.intrinsic_score == 0.9


def test_connect_upward_and_directions():
    engine = NexusEngine()
    c1 = ConceptBlueprint(
        id="matrix-multiplication",
        title="Matrix Multiplication",
        mental_model="Linear transformation composition.",
    )
    c2 = ConceptBlueprint(
        id="neural-network-layer",
        title="Neural Network Layer",
        mental_model="Affine map Wx + b followed by non-linear activation.",
    )
    engine.add_concept(c1)
    engine.add_concept(c2)

    # Wire pointing UP: matrix-multiplication ---> neural-network-layer
    conn = engine.connect(
        source_id="matrix-multiplication",
        target_id="neural-network-layer",
        strength=0.95,
    )

    assert conn.strength == 0.95
    assert len(engine.connections) == 1

    # Looking UP from ingredient
    outgoing = engine.get_outgoing("matrix-multiplication")
    assert len(outgoing) == 1
    assert outgoing[0].target_id == "neural-network-layer"

    # Looking DOWN from higher concept
    incoming = engine.get_incoming("neural-network-layer")
    assert len(incoming) == 1
    assert incoming[0].source_id == "matrix-multiplication"

    # Inbound ingredients (direct top-down pointers)
    ingredients = engine.get_foundational_ingredients("neural-network-layer")
    assert len(ingredients) == 1
    assert ingredients[0].id == "matrix-multiplication"

    # Explicit top-down component retrieval
    components = engine.get_components("neural-network-layer")
    assert len(components) == 1
    assert components[0].source_id == "matrix-multiplication"

    # Higher applications (bottom-up pointers)
    higher = engine.get_higher_applications("matrix-multiplication")
    assert len(higher) == 1
    assert higher[0].id == "neural-network-layer"

    applications = engine.get_applications("matrix-multiplication")
    assert len(applications) == 1
    assert applications[0].target_id == "neural-network-layer"


def test_add_bundle():
    engine = NexusEngine()
    target = ConceptBlueprint(
        id="variational-autoencoders",
        title="Variational Autoencoders",
        topic="Generative Models",
        mental_model="Encoder outputs distribution parameters of q(z|x), optimizing the ELBO.",
    )
    prereqs = [
        ConceptBlueprint(
            id="bayes-theorem",
            title="Bayes' Theorem",
            mental_model="Inversion of conditional probabilities.",
        ),
        ConceptBlueprint(
            id="kl-divergence",
            title="KL Divergence",
            mental_model="Relative entropy between approximate posterior and prior.",
        ),
    ]

    res = engine.add_bundle(target, prereqs)

    assert len(engine.concepts) == 3
    assert "variational-autoencoders" in engine.concepts
    assert "bayes-theorem" in engine.concepts
    assert "kl-divergence" in engine.concepts

    # Check that connections strictly point UP into target
    assert len(engine.connections) == 2
    for conn in engine.connections:
        assert conn.target_id == "variational-autoencoders"
        assert conn.strength == 1.0


def test_stability_formula_diminishing_returns():
    engine = NexusEngine()

    # Leaf node (K = 0)
    leaf = ConceptBlueprint(
        id="frontier-concept",
        title="Frontier Concept",
        mental_model="High-level idea with no higher applications yet.",
        intrinsic_score=0.85,
    )
    engine.add_concept(leaf)

    m_leaf = engine.calculate_concept_stability("frontier-concept")
    assert m_leaf["breadth_k"] == 0
    assert m_leaf["stability"] == 0.85
    assert m_leaf["is_frontier_leaf"] is True

    # Prerequisite concept with multiple outgoing connections
    prereq = ConceptBlueprint(
        id="svd",
        title="Singular Value Decomposition",
        mental_model="Factorization of real matrix into U Sigma V^T.",
    )
    engine.add_concept(prereq)

    # Connect to 3 targets with strengths 1.0, 0.8, 0.9 (avg = 0.9)
    for i, s in enumerate([1.0, 0.8, 0.9]):
        tgt = ConceptBlueprint(id=f"app-{i}", title=f"App {i}", mental_model="...")
        engine.add_concept(tgt)
        engine.connect("svd", f"app-{i}", strength=s)

    m_prereq = engine.calculate_concept_stability("svd")
    assert m_prereq["breadth_k"] == 3
    assert pytest.approx(m_prereq["avg_strength"], 0.01) == 0.9
    expected_bonus = 1.0 + 0.5 * math.log(1.0 + 3)
    expected_stability = 0.9 * expected_bonus
    assert pytest.approx(m_prereq["stability"], 0.01) == expected_stability


def test_get_study_candidates_scans_upward_and_prioritizes_frayed():
    engine = NexusEngine()
    vae = ConceptBlueprint(
        id="vae",
        title="VAE",
        mental_model="Variational autoencoder framework.",
    )
    bayes = ConceptBlueprint(
        id="bayes",
        title="Bayes Theorem",
        mental_model="Belief updating under evidence.",
    )
    kl = ConceptBlueprint(
        id="kl",
        title="KL Divergence",
        mental_model="Relative entropy.",
    )
    engine.add_concept(vae)
    engine.add_concept(bayes)
    engine.add_concept(kl)

    # Wire upward: bayes -> vae, kl -> vae
    engine.connect("bayes", "vae", strength=0.35)
    engine.connect("kl", "vae", strength=0.98)

    candidates = engine.get_study_candidates(top_k=2)

    assert len(candidates) >= 2
    # The first candidate must be 'bayes' because its upward connection to vae is heavily frayed (0.35)
    top = candidates[0]
    assert top.concept.id == "bayes"
    assert top.frayed_target is not None
    assert top.frayed_target.id == "vae"
    assert top.frayed_connection.strength == 0.35


def test_review_joint_and_bundle():
    engine = NexusEngine()
    c1 = ConceptBlueprint(id="c1", title="C1", mental_model="M1")
    c2 = ConceptBlueprint(id="c2", title="C2", mental_model="M2")
    engine.add_concept(c1)
    engine.add_concept(c2)
    engine.connect("c1", "c2", strength=0.5)

    # Review joint
    updated_conn = engine.review_joint("c1", "c2", score=0.9)
    assert updated_conn.strength == 0.9
    assert updated_conn.review_count == 1
    assert updated_conn.last_reviewed is not None

    # Review bundle
    c3 = ConceptBlueprint(id="c3", title="C3", mental_model="M3")
    engine.add_concept(c3)
    engine.connect("c3", "c2", strength=0.4)

    bundle_res = engine.review_bundle("c2", {"c1": 0.95, "c3": 0.85})
    assert bundle_res["updated_joints"] == 2
    assert engine.get_incoming("c2")[0].strength in [0.95, 0.85]
