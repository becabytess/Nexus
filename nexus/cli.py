"""Command Line Interface for Nexus: Directed Cognitive Blueprint & Connection-Strength Engine."""
from __future__ import annotations

import argparse
import json
import os
import sys

from nexus.engine import NexusEngine, slugify
from nexus.models import ConceptBlueprint
from nexus.storage import DEFAULT_GRAPH_FILE, load_graph, save_graph


def _get_engine(path: str = DEFAULT_GRAPH_FILE) -> NexusEngine:
    return load_graph(path)


def cmd_init(args) -> None:
    """Pre-seed the graph with realistic, full-fledged Markdown explainer documents."""
    engine = _get_engine(args.graph)
    if engine.concepts and not args.force:
        print(f"Graph already contains {len(engine.concepts)} concepts. Use --force to overwrite.")
        return

    engine = NexusEngine()

    vae = ConceptBlueprint(
        id="variational-autoencoders",
        title="Variational Autoencoders (VAEs)",
        topic="Deep Learning",
        content="""# Variational Autoencoders (VAEs)

## 1. Problem Formulation & The Density Estimation Bottleneck
In generative modeling, we seek to model an unknown continuous data distribution $p(x)$ over high-dimensional observations $x \in \mathbb{R}^D$. Standard autoencoders compress inputs into deterministic coordinate vectors $z = f_\theta(x)$ inside a low-dimensional latent space $\mathbb{R}^d$. This deterministic mapping creates an uncontrolled latent topology with vast unmapped gaps: sampling an arbitrary latent vector $z \sim \mathcal{N}(0, I)$ and decoding it produces nonsensical outputs because the decoder was never trained on those empty regions.

VAEs solve this by framing generation probabilistically: we introduce continuous latent variables $z$ governed by a prior distribution $p(z) = \mathcal{N}(0, I)$. The marginal data likelihood is:
$$p(x) = \int p(x, z) dz = \int p_\theta(x|z) p(z) dz$$
Direct optimization via Maximum Likelihood Estimation is computationally intractable because the integral requires evaluating all possible latent configurations across high-dimensional space. Furthermore, the true posterior:
$$p(z|x) = \frac{p_\theta(x|z) p(z)}{p(x)}$$
cannot be computed analytically due to the intractable denominator $p(x)$.

## 2. Variational Inference & The Evidence Lower Bound (ELBO)
To circumvent the intractable true posterior, VAEs introduce an inference network (encoder) $q_\phi(z|x)$ that outputs parameters of a tractable Gaussian distribution:
$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \Sigma_\phi(x))$$
where $\Sigma_\phi(x) = \operatorname{diag}(\sigma_\phi^2(x))$.

We measure the discrepancy between the approximation $q_\phi(z|x)$ and the true posterior $p(z|x)$ using Kullback-Leibler (KL) divergence:
$$D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z|x)) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p(z|x)}\right]$$
Expanding this divergence via Bayes' theorem yields:
$$\log p(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z)) + D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z|x))$$
Because KL divergence is strictly non-negative ($D_{\mathrm{KL}} \ge 0$), we obtain the Evidence Lower Bound:
$$\mathcal{L}_{\mathrm{ELBO}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z)) \le \log p(x)$$

The objective decomposes into two opposing forces:
1. **Reconstruction Log-Likelihood**: $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ penalizes reconstruction errors, driving the latent space to retain discriminative features.
2. **Prior Regularization**: $D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z))$ forces every conditional posterior $q_\phi(z|x)$ to conform to the standard normal prior $\mathcal{N}(0, I)$, preventing isolated clustering and ensuring smooth interpolation across the entire manifold.

## 3. The Reparameterization Trick
Backpropagating gradients through the stochastic sampling step $z \sim q_\phi(z|x)$ is impossible with standard automatic differentiation because the node is stochastic. VAEs resolve this by isolating randomness into an independent auxiliary noise variable:
$$\epsilon \sim \mathcal{N}(0, I)$$
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$$
Because the transformation is deterministic and differentiable with respect to $\mu_\phi$ and $\sigma_\phi$, analytical gradients $\nabla_\phi \mathcal{L}_{\mathrm{ELBO}}$ flow end-to-end through the encoder network.

## 4. Analytical KL Divergence for Gaussians
When both the variational posterior $q_\phi(z|x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ and the prior $p(z) = \mathcal{N}(0, I)$ are Gaussian, the KL divergence resolves to closed form:
$$D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^d \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

## 5. Failure Modes & Nuances
- **Posterior Collapse**: When the decoder network $p_\theta(x|z)$ is excessively expressive (e.g., deep autoregressive models), it can model $p(x)$ directly, ignoring $z$. The KL term then drives $q_\phi(z|x) \to p(z)$ completely ($\mu \to 0, \sigma \to 1$), rendering the latent coordinates useless.
- **Blurriness vs Sharpness**: Under Gaussian assumptions for $p_\theta(x|z)$, the reconstruction loss simplifies to Mean Squared Error (MSE), which penalizes high-frequency variance and produces averaged, blurry outputs compared to adversarial losses.
""",
        intrinsic_score=1.0,
    )

    bayes = ConceptBlueprint(
        id="bayes-theorem",
        title="Bayes' Theorem",
        topic="Probability",
        content="""# Bayes' Theorem & Bayesian Inversion

## 1. Core Principle & Mathematical Statement
Bayes' Theorem is the fundamental law of belief updating under observed evidence. For an unobserved parameter or latent state $z$ and an observed dataset $x$:
$$P(z|x) = \frac{P(x|z) P(z)}{P(x)}$$

The components represent distinct probabilistic entities:
- **Prior $P(z)$**: The initial probability distribution over hypotheses before observing evidence.
- **Likelihood $P(x|z)$**: The probability of the observed data conditioned on a specific hypothesis $z$.
- **Evidence / Marginal Likelihood $P(x)$**: The total probability of the data averaged over all possible hypotheses:
  $$P(x) = \int P(x|z) P(z) dz$$
- **Posterior $P(z|x)$**: The updated distribution over hypotheses after incorporating the data $x$.

## 2. The Intractability of the Denominator
In low-dimensional discrete settings, computing $P(x) = \sum_i P(x|z_i) P(z_i)$ is trivial. However, when $z \in \mathbb{R}^d$ is high-dimensional (e.g. latent vectors in deep learning, weight spaces in neural networks), the integral:
$$P(x) = \int_{\mathbb{R}^d} P(x|z) P(z) dz$$
has no closed-form solution for complex non-linear models. This intractability directly necessitates approximate inference methods such as Markov Chain Monte Carlo (MCMC) and Variational Inference.

## 3. Base Rate Fallacy & Concrete Numerical Behavior
A key intuition in Bayesian reasoning is that when the prior probability $P(z)$ of an event is extremely low, even a highly accurate diagnostic test yields a low posterior probability upon a positive test result.
If a disease has prevalence $P(D) = 0.001$, and a test has $99\%$ sensitivity ($P(+|D) = 0.99$) and $95\%$ specificity ($P(-|\neg D) = 0.95$):
$$P(D|+) = \frac{P(+|D) P(D)}{P(+|D) P(D) + P(+|\neg D) P(\neg D)} = \frac{0.99 \times 0.001}{(0.99 \times 0.001) + (0.05 \times 0.999)} \approx 0.0194$$
Even after testing positive, the posterior probability of having the disease is less than $2\%$, because the false-positive volume from the massive healthy population overwhelms the true positives.
""",
        intrinsic_score=1.0,
    )

    kl = ConceptBlueprint(
        id="kl-divergence",
        title="KL Divergence",
        topic="Information Theory",
        content="""# Kullback-Leibler (KL) Divergence

## 1. Definition & Information-Theoretic Foundations
Kullback-Leibler divergence (relative entropy) measures the statistical distance and informational inefficiency when a true probability distribution $P$ is approximated by distribution $Q$.
For continuous random variables with densities $p(x)$ and $q(x)$:
$$D_{\mathrm{KL}}(P \parallel Q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} dx = \mathbb{E}_{x \sim P}\left[\log p(x) - \log q(x)\right]$$

In coding theory, if events occur according to true distribution $P$, the optimal code length for event $x$ is $-\log_2 p(x)$ bits. If we mistakenly construct our code assuming distribution $Q$ with lengths $-\log_2 q(x)$, the expected excess code length per symbol is precisely $D_{\mathrm{KL}}(P \parallel Q)$.

## 2. Invariants & Asymmetry
1. **Non-negativity ($D_{\mathrm{KL}} \ge 0$)**: Proven via Jensen's inequality applied to the convex function $f(t) = t \log t$:
   $$D_{\mathrm{KL}}(P \parallel Q) = \mathbb{E}_{Q}\left[\frac{p(x)}{q(x)} \log \frac{p(x)}{q(x)}\right] \ge f\left(\mathbb{E}_Q\left[\frac{p(x)}{q(x)}\right]\right) = f(1) = 0$$
   Equality holds if and only if $P = Q$ almost everywhere.
2. **Asymmetry ($D_{\mathrm{KL}}(P \parallel Q) \ne D_{\mathrm{KL}}(Q \parallel P)$)**: KL divergence is not a true mathematical distance metric because it violates symmetry and the triangle inequality.

## 3. Forward vs Reverse KL Optimization
- **Forward KL ($D_{\mathrm{KL}}(P \parallel Q)$)**: The expectation is under true distribution $P$. If $p(x) > 0$, then $q(x)$ must be non-zero to prevent the ratio $p(x)/q(x) \to \infty$. This produces **zero-avoiding / mean-seeking** behavior: $Q$ stretches to cover all modes of $P$.
- **Reverse KL ($D_{\mathrm{KL}}(Q \parallel P)$)**: The expectation is under approximate distribution $Q$. If $p(x) = 0$, $q(x)$ must also be 0 to avoid a massive penalty. This produces **zero-forcing / mode-seeking** behavior: $Q$ fits tightly inside a single dominant mode of $P$, ignoring other modes. Variational inference optimizes reverse KL.
""",
        intrinsic_score=1.0,
    )

    bottleneck = ConceptBlueprint(
        id="latent-space-bottleneck",
        title="Latent Space Bottleneck",
        topic="Representation Learning",
        content="""# Latent Space Bottleneck

## 1. Information Bottleneck Principle
A latent space bottleneck is an intentional dimensional constriction introduced into a neural architecture to prevent trivial identity mapping and force representation abstraction.
Given input $X$ and target $Y$, the Information Bottleneck objective maximizes mutual information with the target while minimizing mutual information with the input:
$$\max_{\theta} I(Z; Y) - \beta I(Z; X)$$
where $Z = f_\theta(X)$ is the bottleneck representation.

## 2. Dimensionality vs Manifold Discovery
Real-world continuous data (e.g. natural images in $\mathbb{R}^{D}$ with $D = 10^5$) does not uniformly occupy high-dimensional space; it concentrates near a low-dimensional non-linear sub-manifold $\mathcal{M} \subset \mathbb{R}^D$ of intrinsic dimension $d \ll D$.
Constricting the bottleneck dimension to $d$:
1. Filters high-frequency stochastic sensor noise.
2. Disentangles independent causal coordinates (e.g. azimuth, elevation, illumination).
3. Compresses redundant spatial correlations into orthogonal generative axes.
""",
        intrinsic_score=1.0,
    )

    engine.add_bundle(vae, [bayes, kl, bottleneck])

    bnn = ConceptBlueprint(
        id="bayesian-neural-networks",
        title="Bayesian Neural Networks (BNNs)",
        topic="Machine Learning",
        content="""# Bayesian Neural Networks (BNNs)

## 1. Deterministic vs Probabilistic Weight Spaces
Standard neural networks optimize point estimates of parameter matrices:
$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{i} \mathcal{L}(f(x_i; \mathbf{w}), y_i)$$
This yields overconfident predictions on out-of-distribution inputs because a single weight configuration produces high confidence even far from the training data support.

BNNs place a prior distribution over network weights $p(\mathbf{w}) = \mathcal{N}(0, \sigma_0^2 I)$. Given dataset $\mathcal{D} = \{(x_i, y_i)\}$, learning computes the weight posterior via Bayes' rule:
$$p(\mathbf{w}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{w}) p(\mathbf{w})}{\int p(\mathcal{D}|\mathbf{w}) p(\mathbf{w}) d\mathbf{w}}$$

## 2. Bayes by Backprop (Variational BNNs)
Because the integral over millions of weights is intractable, we fit a variational distribution $q_\theta(\mathbf{w})$ parameterized by mean $\mu$ and variance $\rho$ (where $\sigma = \operatorname{softplus}(\rho)$).
Using the local reparameterization trick:
$$\mathbf{w} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
The network minimizes the variational free energy (negative ELBO):
$$\mathcal{F}(\theta, \mathcal{D}) = D_{\mathrm{KL}}(q_\theta(\mathbf{w}) \parallel p(\mathbf{w})) - \mathbb{E}_{q_\theta(\mathbf{w})}[\log p(\mathcal{D}|\mathbf{w})]$$

## 3. Epistemic vs Aleatoric Uncertainty
- **Aleatoric Uncertainty**: Inherent observational noise in the data generating process (captured by the output distribution variance $\sigma_y^2(x)$).
- **Epistemic Uncertainty**: Ignorance about which model parameters are correct due to limited training data (captured by the variance of predictions across weight samples $\mathbf{w}^{(s)} \sim q_\theta(\mathbf{w})$).
""",
        intrinsic_score=1.0,
    )
    engine.add_concept(bnn)

    engine.connect("variational-autoencoders", "bayesian-neural-networks", strength=0.85)
    engine.connect("bayes-theorem", "bayesian-neural-networks", strength=0.90)

    # Set one intentionally frayed link for study surfacing
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
            "stability_score": c.stability_score,
            "weak_upward_connection": (
                {
                    "target_id": c.frayed_connection.target_id,
                    "target_title": c.frayed_target.title if c.frayed_target else "",
                    "strength": c.frayed_connection.strength,
                }
                if c.frayed_connection
                else None
            ),
            "prerequisites": [
                {"id": ing.id, "title": ing.title} for ing in c.foundational_ingredients
            ],
            "content": c.concept.content,
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
            }
            for conn in outgoing
        ],
        "built_from_below_by": [
            {
                "source_id": conn.source_id,
                "source_title": engine.concepts[conn.source_id].title if conn.source_id in engine.concepts else "",
                "connection_strength": conn.strength,
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
    """Add a single concept with its Markdown explainer document."""
    engine = _get_engine(args.graph)
    content = ""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
    elif args.content:
        content = args.content
    elif args.data:
        data = json.loads(args.data)
        cid = data.get("id") or slugify(data["title"])
        bp = ConceptBlueprint(
            id=cid,
            title=data["title"],
            topic=data.get("topic", "General"),
            content=data["content"],
            intrinsic_score=float(data.get("intrinsic_score", 1.0)),
        )
        engine.add_concept(bp)
        save_graph(engine, args.graph)
        print(json.dumps({"status": "added", "concept_id": bp.id, "title": bp.title}, indent=2))
        return
    else:
        print("Error: Must provide either --content, --file, or --data JSON.", file=sys.stderr)
        sys.exit(1)

    if not args.title:
        print("Error: Must specify --title.", file=sys.stderr)
        sys.exit(1)

    cid = args.id or slugify(args.title)
    bp = ConceptBlueprint(
        id=cid,
        title=args.title,
        topic=args.topic or "General",
        content=content,
        intrinsic_score=float(args.intrinsic_score or 1.0),
    )
    engine.add_concept(bp)
    save_graph(engine, args.graph)
    print(json.dumps({"status": "added", "concept_id": bp.id, "title": bp.title, "topic": bp.topic}, indent=2))


def cmd_add_bundle(args) -> None:
    """Record a learning session: add concept and wire its prerequisites pointing UP to it."""
    engine = _get_engine(args.graph)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif args.data:
        data = json.loads(args.data)
    else:
        print("Error: Must provide either --data JSON string or --file path.", file=sys.stderr)
        sys.exit(1)

    c_data = data.get("concept") or data.get("target")
    if not c_data:
        print("Error: JSON must include 'concept' object.", file=sys.stderr)
        sys.exit(1)

    c_id = c_data.get("id") or slugify(c_data["title"])
    main_bp = ConceptBlueprint(
        id=c_id,
        title=c_data["title"],
        topic=c_data.get("topic", "General"),
        content=c_data["content"],
        intrinsic_score=float(c_data.get("intrinsic_score", 1.0)),
    )

    raw_prereqs = data.get("prerequisites") or data.get("checkpoints") or []
    prereq_bps = []
    for p in raw_prereqs:
        pid = p.get("id") or slugify(p["title"])
        bp = ConceptBlueprint(
            id=pid,
            title=p["title"],
            topic=p.get("topic") or main_bp.topic,
            content=p["content"],
            intrinsic_score=float(p.get("intrinsic_score", 1.0)),
        )
        prereq_bps.append(bp)

    res = engine.add_bundle(main_bp, prereq_bps)
    save_graph(engine, args.graph)

    print(json.dumps({
        "status": "bundle_created",
        "concept_id": res["concept"].id,
        "title": res["concept"].title,
        "prerequisites_created": [p.title for p in res["prerequisites"]],
        "connections_wired": [
            {"from": c.source_id, "to": c.target_id, "strength": c.strength}
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

    # Add Concept
    p_add_c = subparsers.add_parser("add-concept", help="Add a single concept blueprint")
    p_add_c.add_argument("--data", help="JSON string of concept data")
    p_add_c.add_argument("--file", help="Path to a Markdown file containing the concept explainer")
    p_add_c.add_argument("--content", help="Full Markdown explainer text")
    p_add_c.add_argument("--id", help="Optional concept slug ID")
    p_add_c.add_argument("--title", help="Concept title")
    p_add_c.add_argument("--topic", help="Domain or topic")
    p_add_c.add_argument("--intrinsic-score", type=float, default=1.0, help="Intrinsic score [0.0, 1.0]")
    p_add_c.set_defaults(func=cmd_add_concept)

    # Add Bundle
    p_bundle = subparsers.add_parser("add-bundle", help="Add concept and wire prerequisites pointing UP to it")
    p_bundle.add_argument("--data", help="JSON string representing the bundle")
    p_bundle.add_argument("--file", help="Path to JSON file representing the bundle")
    p_bundle.set_defaults(func=cmd_add_bundle)

    # Graph
    p_graph = subparsers.add_parser("graph", help="Print ASCII overview of the directed network")
    p_graph.set_defaults(func=cmd_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
