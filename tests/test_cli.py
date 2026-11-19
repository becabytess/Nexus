"""Portable end-to-end tests for the Nexus CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args, graph_path: str, stdin_text: str | None = None):
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pp}" if existing_pp else REPO_ROOT
    cmd = [sys.executable, "-m", "nexus.cli", "--graph", graph_path] + list(args)
    return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, env=env)


def test_cli_init_preserves_latex_status_inspect_and_diagnose():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")

        res_init = run_cli("init", graph_path=graph_file)
        assert res_init.returncode == 0
        assert "Initialized Nexus graph" in res_init.stdout

        res_status = run_cli("status", graph_path=graph_file)
        assert res_status.returncode == 0
        status_data = json.loads(res_status.stdout)
        assert status_data["concept_count"] == 5
        assert status_data["active_faults"][0]["source_id"] == "bayes-theorem"

        res_vae = run_cli("inspect", "variational-autoencoders", graph_path=graph_file)
        assert res_vae.returncode == 0
        vae_data = json.loads(res_vae.stdout)
        assert r"\frac{p_\theta(x|z)" in vae_data["blueprint"]["content"]
        assert "\x0c" not in vae_data["blueprint"]["content"]

        res_diag = run_cli("diagnose", "bayesian-neural-networks", graph_path=graph_file)
        assert res_diag.returncode == 0
        diag_data = json.loads(res_diag.stdout)
        assert diag_data["weakest_joint"]["source_id"] == "bayes-theorem"
        assert diag_data["weakest_joint"]["target_id"] == "variational-autoencoders"

        res_diag_brief = run_cli("diagnose", "bayesian-neural-networks", "--brief", graph_path=graph_file)
        assert res_diag_brief.returncode == 0
        assert "Fault-Isolation Cone" in res_diag_brief.stdout


def test_cli_list_study_brief_review_modes_and_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")
        run_cli("init", graph_path=graph_file)

        res_list = run_cli("list", "--query", "bayes", graph_path=graph_file)
        assert res_list.returncode == 0
        items = json.loads(res_list.stdout)
        assert any(i["id"] == "bayes-theorem" for i in items)

        res_study = run_cli("study", "--top-k", "1", graph_path=graph_file)
        assert res_study.returncode == 0
        top = json.loads(res_study.stdout)[0]
        assert top["concept_id"] == "bayes-theorem"
        assert top["review_reason"] == "repair"
        assert top["weak_upward_connection"]["strength"] == 0.40
        assert "Reparameterization Trick" in top["weak_upward_connection"]["target_content"]

        res_brief = run_cli("study", "--top-k", "1", "--brief", graph_path=graph_file)
        assert res_brief.returncode == 0
        assert "Socratic Study Target: Bayes' Theorem" in res_brief.stdout

        res_rev = run_cli(
            "review",
            "--source", "bayes-theorem",
            "--target", "variational-autoencoders",
            "--score", "0.95",
            graph_path=graph_file,
        )
        assert res_rev.returncode == 0
        rev_data = json.loads(res_rev.stdout)
        assert rev_data["status"] == "updated"
        assert rev_data["new_strength"] == 0.95

        res_leaf = run_cli(
            "review",
            "--concept", "bayesian-neural-networks",
            "--score", "1.0",
            graph_path=graph_file,
        )
        assert res_leaf.returncode == 0
        leaf_data = json.loads(res_leaf.stdout)
        assert leaf_data["mode"] == "concept"
        assert len(leaf_data["implicitly_reinforced"]) >= 2

        res_bundle = run_cli(
            "review",
            "--target", "variational-autoencoders",
            "--scores", json.dumps({"bayes-theorem": 0.5, "kl-divergence": 0.98}),
            graph_path=graph_file,
        )
        assert res_bundle.returncode == 0
        assert json.loads(res_bundle.stdout)["updated_joints"] == 2

        wiki_dir = os.path.join(tmpdir, "wiki")
        res_exp = run_cli("export", wiki_dir, graph_path=graph_file)
        assert res_exp.returncode == 0
        assert json.loads(res_exp.stdout)["files_written"] == 6
        assert os.path.exists(os.path.join(wiki_dir, "INDEX.md"))


def test_cli_add_connect_remove_and_stdin_bundle():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")

        res_c = run_cli(
            "add-concept",
            "--title", "Eigenvalues",
            "--topic", "Linear Algebra",
            "--file", "-",
            graph_path=graph_file,
            stdin_text="# Eigenvalues\n\n$Av = \\lambda v$.",
        )
        assert res_c.returncode == 0
        assert json.loads(res_c.stdout)["concept_id"] == "eigenvalues"

        bundle_json = json.dumps({
            "concept": {
                "title": "Principal Component Analysis",
                "topic": "Machine Learning",
                "content": "# PCA\n\nProjection onto top eigenvectors of the covariance matrix.",
            },
            "prerequisites": [
                "eigenvalues",
                {
                    "title": "Covariance Matrix",
                    "topic": "Linear Algebra",
                    "content": "# Covariance Matrix\n\nPairwise covariance structure.",
                },
            ],
        })
        res_b = run_cli("add-bundle", "--file", "-", graph_path=graph_file, stdin_text=bundle_json)
        assert res_b.returncode == 0
        b_data = json.loads(res_b.stdout)
        assert b_data["concept_id"] == "principal-component-analysis"
        assert len(b_data["connections_wired"]) == 2

        res_cyc = run_cli(
            "connect",
            "--source", "principal-component-analysis",
            "--target", "eigenvalues",
            graph_path=graph_file,
        )
        assert res_cyc.returncode == 1

        res_disc = run_cli(
            "remove",
            "--source", "eigenvalues",
            "--target", "principal-component-analysis",
            graph_path=graph_file,
        )
        assert res_disc.returncode == 0
        assert json.loads(res_disc.stdout)["status"] == "disconnected"

        res_rem = run_cli("remove", "eigenvalues", graph_path=graph_file)
        assert res_rem.returncode == 0
        assert json.loads(res_rem.stdout)["status"] == "removed"


def test_cli_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")
        run_cli("init", graph_path=graph_file)

        res_g = run_cli("graph", graph_path=graph_file)
        assert res_g.returncode == 0
        assert "NEXUS DIRECTED COGNITIVE NETWORK" in res_g.stdout
        assert "LAYER 0: FOUNDATIONS" in res_g.stdout
        assert "FRONTIER" in res_g.stdout