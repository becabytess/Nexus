"""Tests for Nexus graph engine: DAG safety, decay, spacing, dual urgency, diagnostics, and FIRe."""
from datetime import datetime, timedelta, timezone
import pytest

from nexus.engine import NexusEngine, breadth_multiplier, slugify
from nexus.models import ConceptBlueprint


def _t(days: float = 0.0) -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(days=days)


def test_slugify():
    assert slugify("Variational Autoencoders (VAEs)") == "variational-autoencoders-vaes"
    assert slugify("Bayes' Theorem") == "bayes-theorem"
    assert slugify("  Linear_Algebra   101 ") == "linear-algebra-101"
    with pytest.raises(ValueError):
        slugify("???")


def test_add_update_remove_and_disconnect():
    engine = NexusEngine()
    bp1 = ConceptBlueprint(id="a", title="A", topic="Math", content="# A", intrinsic_score=0.9)
    bp2 = ConceptBlueprint(id="b", title="B", topic="Math", content="# B")
    engine.add_concept(bp1)
    engine.add_concept(bp2)

    orig_created = engine.concepts["a"].created_at
    engine.add_concept(ConceptBlueprint(id="a", title="A Revised", topic="Math", content="# A v2"))
    assert engine.concepts["a"].title == "A Revised"
    assert engine.concepts["a"].created_at == orig_created

    engine.connect("a", "b", strength=0.8)
    assert len(engine.connections) == 1

    engine.disconnect("a", "b")
    assert len(engine.connections) == 0
    assert engine.get_outgoing("a") == []
    assert engine.get_incoming("b") == []

    engine.connect("a", "b", strength=0.8)
    engine.remove_concept("a")
    assert "a" not in engine.concepts
    assert len(engine.connections) == 0
    assert engine.get_incoming("b") == []


def test_connect_rejects_self_loops_and_cycles():
    engine = NexusEngine()
    for cid in ("a", "b", "c"):
        engine.add_concept(ConceptBlueprint(id=cid, title=cid.upper(), content=f"# {cid}"))

    with pytest.raises(ValueError, match="itself"):
        engine.connect("a", "a")

    engine.connect("a", "b")
    engine.connect("b", "c")

    updated = engine.connect("a", "b", strength=0.6)
    assert updated.strength == 0.6

    with pytest.raises(ValueError, match="cycle"):
        engine.connect("c", "a")
    with pytest.raises(ValueError, match="cycle"):
        engine.connect("b", "a")


def test_add_bundle_reuses_existing_and_preserves_frayed_links():
    engine = NexusEngine()
    t0 = _t(0)
    vae = ConceptBlueprint(id="vae", title="VAE", topic="DL", content="# VAE")
    bayes = ConceptBlueprint(id="bayes", title="Bayes", topic="Prob", content="# Bayes\n\nDetailed explainer.")
    kl = ConceptBlueprint(id="kl", title="KL", topic="Info", content="# KL")

    engine.add_bundle(vae, [bayes, kl])
    engine.review_joint("bayes", "vae", score=0.40, now=t0)

    bottleneck = ConceptBlueprint(id="bottleneck", title="Bottleneck", content="# Bottleneck")
    res = engine.add_bundle(
        ConceptBlueprint(id="vae", title="VAE", topic="DL", content="# VAE\n\nExpanded."),
        ["bayes", bottleneck],
    )

    assert len(res["prerequisites"]) == 2
    assert engine.get_connection("bayes", "vae").strength == 0.40
    assert engine.get_connection("bottleneck", "vae").strength == 1.0
    assert "Detailed explainer." in engine.concepts["bayes"].content


def test_stability_formula_and_breadth_anchored_decay():
    engine = NexusEngine()
    t0 = _t(0)

    leaf = ConceptBlueprint(
        id="leaf", title="Leaf", content="# Leaf", intrinsic_score=0.85, created_at=t0.isoformat()
    )
    engine.add_concept(leaf)
    m_leaf = engine.calculate_concept_stability("leaf", now=t0)
    assert m_leaf["breadth_k"] == 0
    assert m_leaf["stability"] == 0.85
    assert m_leaf["is_frontier_leaf"] is True

    broad = ConceptBlueprint(id="svd", title="SVD", content="# SVD", created_at=t0.isoformat())
    narrow = ConceptBlueprint(id="narrow", title="Narrow", content="# Narrow", created_at=t0.isoformat())
    engine.add_concept(broad)
    engine.add_concept(narrow)

    for i, s in enumerate([1.0, 0.8, 0.9]):
        tgt = ConceptBlueprint(id=f"app-{i}", title=f"App {i}", content="# App", created_at=t0.isoformat())
        engine.add_concept(tgt)
        engine.connect("svd", f"app-{i}", strength=s, now=t0)
    engine.connect("narrow", "app-0", strength=1.0, now=t0)

    m_broad = engine.calculate_concept_stability("svd", now=t0)
    assert pytest.approx(m_broad["stability"], abs=1e-3) == 0.9 * breadth_multiplier(3)

    t2 = _t(2)
    eff_broad = engine.effective_connection_strength(engine.get_connection("svd", "app-0"), now=t2)
    eff_narrow = engine.effective_connection_strength(engine.get_connection("narrow", "app-0"), now=t2)
    assert eff_broad > eff_narrow
    assert eff_narrow < 1.0


def test_spacing_effect_is_cram_proof_and_failure_contracts():
    engine = NexusEngine()
    t0 = _t(0)
    engine.add_concept(ConceptBlueprint(id="u", title="U", content="# U", created_at=t0.isoformat()))
    engine.add_concept(ConceptBlueprint(id="v", title="V", content="# V", created_at=t0.isoformat()))
    engine.connect("u", "v", strength=1.0, now=t0)

    r_cram = engine.review_joint("u", "v", score=1.0, now=t0)
    assert r_cram.new_half_life == 1.0

    t_spaced = _t(1.5)
    r_spaced = engine.review_joint("u", "v", score=1.0, now=t_spaced)
    assert r_spaced.new_half_life > 1.8

    r_fail = engine.review_joint("u", "v", score=0.0, now=t_spaced)
    assert r_fail.new_half_life == pytest.approx(r_spaced.new_half_life * 0.5, abs=1e-3)


def test_asymmetric_implicit_confirmation_on_diamond_dag():
    engine = NexusEngine()
    t0 = _t(0)
    for cid in ("bayes", "kl", "vae", "bnn"):
        engine.add_concept(ConceptBlueprint(id=cid, title=cid.upper(), content=f"# {cid}", created_at=t0.isoformat()))

    engine.connect("bayes", "vae", strength=0.60, now=t0)
    engine.connect("kl", "vae", strength=0.70, now=t0)
    engine.connect("vae", "bnn", strength=0.80, now=t0)
    engine.connect("bayes", "bnn", strength=0.85, now=t0)

    fail_res = engine.review_concept("bnn", score=0.20, now=t0)
    assert fail_res.implicitly_reinforced == []
    assert engine.get_connection("vae", "bnn").strength == 0.80
    assert engine.get_connection("bayes", "vae").strength == 0.60

    pass_res = engine.review_concept("bnn", score=1.0, now=t0)
    by_edge = {(ir.source_id, ir.target_id): ir for ir in pass_res.implicitly_reinforced}

    assert set(by_edge.keys()) == {
        ("vae", "bnn"),
        ("bayes", "bnn"),
        ("bayes", "vae"),
        ("kl", "vae"),
    }
    assert by_edge[("vae", "bnn")].depth == 1
    assert by_edge[("vae", "bnn")].new_strength == pytest.approx(0.90, abs=1e-3)
    assert by_edge[("bayes", "bnn")].depth == 1
    assert by_edge[("bayes", "bnn")].new_strength == pytest.approx(0.925, abs=1e-3)
    assert by_edge[("bayes", "vae")].depth == 2
    assert by_edge[("bayes", "vae")].new_strength == pytest.approx(0.70, abs=1e-3)
    assert by_edge[("kl", "vae")].depth == 2
    assert by_edge[("kl", "vae")].new_strength == pytest.approx(0.775, abs=1e-3)


def test_dual_urgency_edge_level_repair_and_top_down_leverage():
    engine = NexusEngine()
    t0 = _t(0)
    for cid in ("bayes", "kl", "vae", "bnn"):
        engine.add_concept(ConceptBlueprint(id=cid, title=cid.upper(), content=f"# {cid}", created_at=t0.isoformat()))

    # Two frayed outgoing joints from the SAME concept ('bayes') must BOTH surface
    engine.connect("bayes", "vae", strength=0.30, now=t0)
    engine.connect("bayes", "bnn", strength=0.35, now=t0)
    engine.connect("kl", "vae", strength=1.0, now=t0)
    engine.connect("vae", "bnn", strength=1.0, now=t0)

    cands_repair = engine.get_study_candidates(top_k=2, now=t0)
    assert (cands_repair[0].concept.id, cands_repair[0].frayed_target.id) == ("bayes", "vae")
    assert (cands_repair[1].concept.id, cands_repair[1].frayed_target.id) == ("bayes", "bnn")
    assert cands_repair[0].review_reason == "repair"

    # Once mastered (all 1.0), after 4 days of decay, the frontier leaf ('bnn') rises to #1
    engine.connect("bayes", "vae", strength=1.0, now=t0)
    engine.connect("bayes", "bnn", strength=1.0, now=t0)
    t4 = _t(4)
    cands_leverage = engine.get_study_candidates(top_k=1, now=t4)
    top = cands_leverage[0]
    assert top.concept.id == "bnn"
    assert top.frayed_connection is None
    assert top.review_reason == "leverage"
    assert top.decay_leverage > top.fault_score
    assert len(top.implicit_refresh_targets) == 4


def test_topological_layers_diagnose_summary_and_search():
    engine = NexusEngine()
    t0 = _t(0)
    engine.add_concept(ConceptBlueprint(id="bayes", title="Bayes' Theorem", topic="Probability", content="# Bayes", created_at=t0.isoformat()))
    engine.add_concept(ConceptBlueprint(id="vae", title="VAEs", topic="Deep Learning", content="# ELBO", created_at=t0.isoformat()))
    engine.add_concept(ConceptBlueprint(id="bnn", title="BNNs", topic="Machine Learning", content="# Weights", created_at=t0.isoformat()))
    engine.add_concept(ConceptBlueprint(id="island", title="Orphan", topic="Misc", content="# Island", created_at=t0.isoformat()))

    engine.connect("bayes", "vae", strength=0.35, now=t0)
    engine.connect("vae", "bnn", strength=0.90, now=t0)
    engine.connect("bayes", "bnn", strength=0.95, now=t0)

    assert engine.topological_layers() == {"bayes": 0, "island": 0, "vae": 1, "bnn": 2}

    diag = engine.diagnose_concept("bnn", now=t0)
    assert diag["weakest_joint"]["source_id"] == "bayes"
    assert diag["weakest_joint"]["target_id"] == "vae"
    assert diag["weakest_joint"]["hops_below"] == 2
    assert len(diag["supporting_cone"]) == 3

    snap = engine.summary(now=t0)
    assert snap["concept_count"] == 4
    assert snap["connection_count"] == 3
    assert snap["keystones"][0]["id"] == "bayes"
    assert snap["active_faults"][0]["source_id"] == "bayes"
    assert [i["id"] for i in snap["isolated_concepts"]] == ["island"]

    assert [c.id for c in engine.search_concepts(query="elbo")] == ["vae"]
    assert [c.id for c in engine.search_concepts(topic="prob")] == ["bayes"]


def test_review_bundle():
    engine = NexusEngine()
    for cid in ("c1", "c2", "c3"):
        engine.add_concept(ConceptBlueprint(id=cid, title=cid.upper(), content=f"# {cid}"))
    engine.connect("c1", "c2", strength=0.5)
    engine.connect("c3", "c2", strength=0.4)

    res = engine.review_bundle("c2", {"c1": 0.95, "c3": 0.30, "nonexistent": 0.5})
    assert res["updated_joints"] == 2
    assert res["skipped_sources"] == ["nonexistent"]
    assert engine.get_connection("c1", "c2").strength == 0.95
    assert engine.get_connection("c3", "c2").strength == 0.30