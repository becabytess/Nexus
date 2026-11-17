"""Tests for the Nexus CLI."""
import json
import os
import subprocess
import sys
import tempfile


def run_cli(*args, graph_path: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = "c:\\Users\\beca\\Desktop\\gravn\\nexus"
    cmd = [sys.executable, "-m", "nexus.cli", "--graph", graph_path] + list(args)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return res


def test_cli_init_and_inspect():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")

        # 1. Run init
        res_init = run_cli("init", graph_path=graph_file)
        assert res_init.returncode == 0
        assert "Initialized Nexus graph" in res_init.stdout

        # 2. Run inspect
        res_insp = run_cli("inspect", "bayes-theorem", graph_path=graph_file)
        assert res_insp.returncode == 0
        data = json.loads(res_insp.stdout)
        assert data["blueprint"]["id"] == "bayes-theorem"
        assert len(data["points_up_into"]) >= 1


def test_cli_study_and_review():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")
        run_cli("init", graph_path=graph_file)

        # Study
        res_study = run_cli("study", "--top-k", "2", graph_path=graph_file)
        assert res_study.returncode == 0
        candidates = json.loads(res_study.stdout)
        assert len(candidates) >= 1
        top_cand = candidates[0]
        assert top_cand["concept_id"] == "bayes-theorem"
        assert top_cand["frayed_joint"]["strength"] == 0.40

        # Review
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


def test_cli_add_concept_and_bundle():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")

        # Add single concept
        res_c = run_cli(
            "add-concept",
            "--title", "Eigenvalues",
            "--mental-model", "Scaling factors along invariant directional eigenvectors Av = lambda v.",
            "--topic", "Linear Algebra",
            graph_path=graph_file,
        )
        assert res_c.returncode == 0
        c_data = json.loads(res_c.stdout)
        assert c_data["concept_id"] == "eigenvalues"

        # Add bundle
        bundle_json = json.dumps({
            "target": {
                "title": "Principal Component Analysis",
                "topic": "Machine Learning",
                "mental_model": "Orthogonal projection onto directions of maximum variance.",
            },
            "checkpoints": [
                {
                    "title": "Covariance Matrix",
                    "mental_model": "Matrix capturing pairwise variances and covariances.",
                    "context_role": "Used to compute eigenvectors and eigenvalues.",
                }
            ],
        })
        res_b = run_cli("add-bundle", "--data", bundle_json, graph_path=graph_file)
        assert res_b.returncode == 0
        b_data = json.loads(res_b.stdout)
        assert b_data["target_id"] == "principal-component-analysis"
        assert "Covariance Matrix" in b_data["checkpoints_created"]


def test_cli_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_file = os.path.join(tmpdir, "graph.json")
        run_cli("init", graph_path=graph_file)

        res_g = run_cli("graph", graph_path=graph_file)
        assert res_g.returncode == 0
        assert "NEXUS DIRECTED COGNITIVE NETWORK" in res_g.stdout
