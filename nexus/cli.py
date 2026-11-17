"""Command Line Interface for Nexus: Directed Cognitive Blueprint & Connection-Strength Engine."""
from __future__ import annotations

import argparse
import json
import os
import sys

from nexus.engine import NexusEngine, slugify
from nexus.models import CheckpointSpec, ConceptBlueprint
from nexus.storage import DEFAULT_GRAPH_FILE, load_graph, save_graph


def _get_engine(path: str = DEFAULT_GRAPH_FILE) -> NexusEngine:
    return load_graph(path)


def cmd_init(args) -> None:
    """Pre-seed the graph with a rich, realistic knowledge network."""
    engine = _get_engine(args.graph)
    if engine.concepts and not args.force:
        print(f"Graph already contains {len(engine.concepts)} concepts. Use --force to overwrite.")
        return

    engine = NexusEngine()

    # Pre-seed: Variational Autoencoders with rich blueprints
    vae = ConceptBlueprint(
        id="variational-autoencoders",
        title="Variational Autoencoders (VAEs)",
        topic="Deep Learning",
        mental_model=(
            "A probabilistic generative framework that models continuous data distributions p(x). "
            "Instead of an encoder compressing an input into a deterministic code vector (which leads to empty "
            "regions in latent space where the decoder generates garbage), the encoder outputs distribution "
            "parameters (mean mu and log-variance log_sigma_sq) of a Gaussian posterior q(z|x). The latent vector "
            "z is sampled via the reparameterization trick z = mu + sigma * epsilon, allowing end-to-end backprop. "
            "The objective maximizes the Evidence Lower Bound (ELBO): reconstruction log-likelihood minus "
            "KL divergence regularization pulling q(z|x) toward a standard normal prior p(z) = N(0, I)."
        ),
        anchor_example=(
            "Deriving the ELBO on paper: log p(x) >= E_q[log p(x|z)] - D_KL(q(z|x) || p(z)). "
            "Reconstruction forces the latent space to retain image features; KL divergence prevents the encoder "
            "from scattering each image into an isolated point, ensuring smooth interpolation."
        ),
        pitfalls=(
            "Confusing VAE with a deterministic autoencoder with noise. The critical difference is the explicit "
            "Bayesian prior regularization that makes the latent space smoothly generative."
        ),
        intrinsic_score=1.0,
    )

    checkpoints = [
        CheckpointSpec(
            title="Bayes' Theorem",
            topic="Probability",
            mental_model=(
                "Belief re-weighting under evidence: P(Hypothesis | Data) = [P(Data | Hypothesis) * P(Hypothesis)] / P(Data). "
                "The posterior is the prior distribution re-weighted by how well each hypothesis explains the observed data. "
                "The denominator P(Data) = integral P(Data|z)P(z)dz is the marginal evidence, which becomes intractable in high dimensions."
            ),
            anchor_example="Medical testing: high false-positive rates when the prior prevalence is low.",
            pitfalls="Confusing likelihood P(Data|Hypothesis) with posterior probability P(Hypothesis|Data).",
            context_role="Formulates the true intractable posterior p(z|x) = p(x|z)p(z)/p(x) that VAE approximates with q(z|x).",
        ),
        CheckpointSpec(
            title="KL Divergence",
            topic="Information Theory",
            mental_model=(
                "A directed, asymmetric measure of statistical distance from distribution Q to reference distribution P: "
                "D_KL(Q || P) = sum Q(x) log(Q(x) / P(x)). It measures the expected excess code length or surprise when "
                "coding events using an approximate model Q instead of true P. It is strictly >= 0 via Jensen's inequality."
            ),
            anchor_example="For two univariate Gaussians N(mu, sigma^2) and N(0, 1), KL divergence has an exact closed form: 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2)).",
            pitfalls="Assuming KL divergence is a metric. It violates symmetry (D_KL(P||Q) != D_KL(Q||P)) and the triangle inequality.",
            context_role="Acts as the regularizer in the ELBO loss, penalizing encoder distributions that drift away from prior N(0, I).",
        ),
        CheckpointSpec(
            title="Latent Space Bottleneck",
            topic="Representation Learning",
            mental_model=(
                "An informational constriction that forces a model to discover low-dimensional coordinate systems "
                "capturing independent generative factors rather than copying raw sensory input."
            ),
            anchor_example="Compressing 784-pixel MNIST digits into a 2D coordinate space (angle and thickness).",
            pitfalls="Assuming smaller latent dimensions are always better; too small causes underfitting/information loss.",
            context_role="Provides the low-dimensional manifold z on which the generative process operates.",
        ),
    ]

    engine.add_bundle(vae, checkpoints)

    # Add a higher-level concept: Bayesian Neural Networks (BNNs) building on VAE
    bnn = ConceptBlueprint(
        id="bayesian-neural-networks",
        title="Bayesian Neural Networks (BNNs)",
        topic="Machine Learning",
        mental_model=(
            "Neural networks where weights are probability distributions rather than static scalar parameters. "
            "Variational inference is used to learn a variational distribution q(w) over the network weights to approximate "
            "the true posterior p(w|Data), naturally providing epistemic uncertainty estimates."
        ),
        anchor_example="Blundell et al. (Weight Uncertainty in Neural Networks / Bayes by Backprop).",
        pitfalls="Confusing aleatoric (data noise) with epistemic (model weight) uncertainty.",
        intrinsic_score=1.0,
    )
    engine.add_concept(bnn)

    # Wire VAE and Bayes pointing UP into Bayesian Neural Networks
    engine.connect(
        source_id="variational-autoencoders",
        target_id="bayesian-neural-networks",
        strength=0.85,
        context_role="Provides the variational approximation framework (ELBO) applied to network weights.",
    )
    engine.connect(
        source_id="bayes-theorem",
        target_id="bayesian-neural-networks",
        strength=0.90,
        context_role="Formulates weight posterior p(w|D) = p(D|w)p(w)/p(D).",
    )

    # Set one deliberately frayed connection to demonstrate study surfacing
    # For instance: Bayes' Theorem connection into VAE is frayed!
    engine.review_joint("bayes-theorem", "variational-autoencoders", score=0.40)

    save_graph(engine, args.graph)
    print(f"Initialized Nexus graph at {args.graph} with {len(engine.concepts)} concepts and {len(engine.connections)} connections.")


def cmd_study(args) -> None:
    """Surface the most urgently needed concept to study by scanning UPWARD."""
    engine = _get_engine(args.graph)
    if not engine.concepts:
        print("Graph is empty. Run 'python -m nexus.cli init' to initialize.")
        return

    candidates = engine.get_study_candidates(top_k=args.top_k)
    output = []
    for c in candidates:
        item = {
            "concept_id": c.concept.id,
            "title": c.concept.title,
            "topic": c.concept.topic,
            "urgency_score": c.urgency_score,
            "reason": c.reason,
            "frayed_joint": (
                {
                    "source": c.frayed_connection.source_id,
                    "target": c.frayed_connection.target_id,
                    "target_title": c.frayed_target.title if c.frayed_target else "",
                    "strength": c.frayed_connection.strength,
                    "context_role": c.frayed_connection.context_role,
                }
                if c.frayed_connection
                else None
            ),
            "foundational_ingredients": [
                {"id": ing.id, "title": ing.title} for ing in c.foundational_ingredients
            ],
            "mental_model": c.concept.mental_model,
            "anchor_example": c.concept.anchor_example,
        }
        output.append(item)

    print(json.dumps(output, indent=2))


def cmd_inspect(args) -> None:
    """Print the complete cognitive blueprint and connections for a concept."""
    engine = _get_engine(args.graph)
    concept = engine.concepts.get(args.concept_id)
    if not concept:
        print(f"Error: Concept {args.concept_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    outgoing = engine.get_outgoing(args.concept_id)
    incoming = engine.get_incoming(args.concept_id)
    stability = engine.calculate_concept_stability(args.concept_id)

    res = {
        "blueprint": concept.model_dump(),
        "stability": stability,
        "points_up_into": [
            {
                "target_id": conn.target_id,
                "target_title": engine.concepts[conn.target_id].title if conn.target_id in engine.concepts else "",
                "connection_strength": conn.strength,
                "context_role": conn.context_role,
            }
            for conn in outgoing
        ],
        "built_from_below_by": [
            {
                "source_id": conn.source_id,
                "source_title": engine.concepts[conn.source_id].title if conn.source_id in engine.concepts else "",
                "connection_strength": conn.strength,
                "context_role": conn.context_role,
            }
            for conn in incoming
        ],
    }
    print(json.dumps(res, indent=2))


def cmd_review(args) -> None:
    """Update connection strength for a specific joint (prerequisite -> target)."""
    engine = _get_engine(args.graph)
    try:
        conn = engine.review_joint(args.source, args.target, score=args.score)
        save_graph(engine, args.graph)
        print(json.dumps({
            "status": "updated",
            "source_id": conn.source_id,
            "target_id": conn.target_id,
            "new_strength": conn.strength,
            "review_count": conn.review_count,
        }, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_add_concept(args) -> None:
    """Add a single rich concept blueprint to the graph."""
    engine = _get_engine(args.graph)
    if args.data:
        data = json.loads(args.data)
        cid = data.get("id") or slugify(data["title"])
        bp = ConceptBlueprint(
            id=cid,
            title=data["title"],
            topic=data.get("topic", "General"),
            mental_model=data["mental_model"],
            anchor_example=data.get("anchor_example", ""),
            pitfalls=data.get("pitfalls", ""),
            intrinsic_score=float(data.get("intrinsic_score", 1.0)),
        )
    elif args.title and args.mental_model:
        cid = args.id or slugify(args.title)
        bp = ConceptBlueprint(
            id=cid,
            title=args.title,
            topic=args.topic or "General",
            mental_model=args.mental_model,
            anchor_example=args.anchor_example or "",
            pitfalls=args.pitfalls or "",
            intrinsic_score=float(args.intrinsic_score or 1.0),
        )
    else:
        print("Error: Must provide either --data JSON or both --title and --mental-model.", file=sys.stderr)
        sys.exit(1)

    engine.add_concept(bp)
    save_graph(engine, args.graph)
    print(json.dumps({
        "status": "added",
        "concept_id": bp.id,
        "title": bp.title,
        "topic": bp.topic,
    }, indent=2))


def cmd_add_bundle(args) -> None:
    """Record an 'aha!' learning session: add target concept and wire checkpoints pointing UP to it."""
    engine = _get_engine(args.graph)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.data:
        data = json.loads(args.data)
    else:
        print("Error: Must provide either --data JSON string or --file path.", file=sys.stderr)
        sys.exit(1)

    target_data = data["target"]
    target_id = target_data.get("id") or slugify(target_data["title"])
    target_bp = ConceptBlueprint(
        id=target_id,
        title=target_data["title"],
        topic=target_data.get("topic", "General"),
        mental_model=target_data["mental_model"],
        anchor_example=target_data.get("anchor_example", ""),
        pitfalls=target_data.get("pitfalls", ""),
        intrinsic_score=float(target_data.get("intrinsic_score", 1.0)),
    )

    checkpoints = [
        CheckpointSpec(
            title=cp["title"],
            topic=cp.get("topic"),
            mental_model=cp["mental_model"],
            anchor_example=cp.get("anchor_example", ""),
            pitfalls=cp.get("pitfalls", ""),
            context_role=cp.get("context_role", "prerequisite ingredient"),
        )
        for cp in data.get("checkpoints", [])
    ]

    res = engine.add_bundle(target_bp, checkpoints)
    save_graph(engine, args.graph)

    print(json.dumps({
        "status": "bundle_created",
        "target_id": res["target"].id,
        "target_title": res["target"].title,
        "checkpoints_created": [cp.title for cp in res["checkpoints"]],
        "connections_wired": [
            {"from": c.source_id, "to": c.target_id, "role": c.context_role}
            for c in res["connections"]
        ],
    }, indent=2))


def cmd_graph(args) -> None:
    """Print an ASCII visualization of the directed cognitive network."""
    engine = _get_engine(args.graph)
    if not engine.concepts:
        print("Empty graph.")
        return

    print("================================================================================")
    print("                     NEXUS DIRECTED COGNITIVE NETWORK                           ")
    print("================================================================================")
    for cid, c in sorted(engine.concepts.items()):
        outgoing = engine.get_outgoing(cid)
        incoming = engine.get_incoming(cid)
        metrics = engine.calculate_concept_stability(cid)

        print(f"\n[NODE] {c.title} ({c.topic}) [{cid}]")
        print(f"       Stability: {metrics['stability']:.2f} | Outgoing Contexts: {metrics['breadth_k']}")
        if outgoing:
            print("       POINTS UP INTO (Outgoing Contexts):")
            for out in outgoing:
                tgt_title = engine.concepts[out.target_id].title if out.target_id in engine.concepts else out.target_id
                bar = "█" * int(out.strength * 10) + "░" * (10 - int(out.strength * 10))
                print(f"         └───[{bar} {out.strength:.2f}]──► {tgt_title}")
                if out.context_role:
                    print(f"              Role: {out.context_role}")
        else:
            print("       └── (Frontier Leaf Node: Top of current knowledge horizon)")

        if incoming:
            print("       BUILT FROM BELOW BY (Foundational Ingredients):")
            for inc in incoming:
                src_title = engine.concepts[inc.source_id].title if inc.source_id in engine.concepts else inc.source_id
                print(f"         ├── {src_title} (Joint Health: {inc.strength:.2f})")
    print("\n================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine")
    parser.add_argument("--graph", default=DEFAULT_GRAPH_FILE, help="Path to graph JSON file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init
    p_init = subparsers.add_parser("init", help="Initialize or pre-seed knowledge graph")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing graph")
    p_init.set_defaults(func=cmd_init)

    # Study
    p_study = subparsers.add_parser("study", help="Scan UPWARD and surface concepts needing review")
    p_study.add_argument("--top-k", type=int, default=3, help="Number of candidates to surface")
    p_study.set_defaults(func=cmd_study)

    # Inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a concept blueprint and its connections")
    p_inspect.add_argument("concept_id", help="Concept identifier slug")
    p_inspect.set_defaults(func=cmd_inspect)

    # Review
    p_review = subparsers.add_parser("review", help="Update the health score of a specific joint")
    p_review.add_argument("--source", required=True, help="Prerequisite concept slug")
    p_review.add_argument("--target", required=True, help="Higher-level concept slug")
    p_review.add_argument("--score", type=float, required=True, help="Health score in [0.0, 1.0]")
    p_review.set_defaults(func=cmd_review)

    # Graph
    p_graph = subparsers.add_parser("graph", help="Print ASCII overview of the directed network")
    p_graph.set_defaults(func=cmd_graph)

    # Add Concept
    p_add_c = subparsers.add_parser("add-concept", help="Add a single concept blueprint")
    p_add_c.add_argument("--data", help="JSON string of concept data")
    p_add_c.add_argument("--id", help="Optional concept slug ID")
    p_add_c.add_argument("--title", help="Concept title")
    p_add_c.add_argument("--topic", help="Domain or topic")
    p_add_c.add_argument("--mental-model", help="Core mechanism explanation")
    p_add_c.add_argument("--anchor-example", help="Concrete anchor example")
    p_add_c.add_argument("--pitfalls", help="Common pitfalls")
    p_add_c.add_argument("--intrinsic-score", type=float, default=1.0, help="Intrinsic score [0.0, 1.0]")
    p_add_c.set_defaults(func=cmd_add_concept)

    # Add Bundle
    p_bundle = subparsers.add_parser("add-bundle", help="Add target concept and wire checkpoints pointing UP")
    p_bundle.add_argument("--data", help="JSON string representing the bundle")
    p_bundle.add_argument("--file", help="Path to JSON file representing the bundle")
    p_bundle.set_defaults(func=cmd_add_bundle)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
