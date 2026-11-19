"""Command Line Interface for Nexus: Directed Cognitive Blueprint & Connection-Strength Engine."""
from __future__ import annotations

import argparse
import json
import sys

from nexus.engine import NexusEngine, slugify
from nexus.models import ConceptBlueprint
from nexus.storage import DEFAULT_GRAPH_FILE, export_markdown, load_graph, save_graph
from nexus.tutor import SocraticTutor


def _get_engine(path: str = DEFAULT_GRAPH_FILE) -> NexusEngine:
    return load_graph(path)


def _read_text_arg(file_path: str) -> str:
    if file_path == "-":
        return sys.stdin.read()
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def cmd_init(args) -> None:
    """Pre-seed the graph with full-fledged Markdown explainer documents."""
    engine = _get_engine(args.graph)
    if engine.concepts and not args.force:
        print(f"Graph already contains {len(engine.concepts)} concepts. Use --force to overwrite.")
        return

    engine = NexusEngine()

    vae = ConceptBlueprint(
        id="variational-autoencoders",
        title="Variational Autoencoders (VAEs)",
        topic="Deep Learning",
        content=r"""# Variational Autoencoders (VAEs)

## 1. Problem Formulation & The Density Estimation Bottleneck
In generative modeling, we seek to model an unknown continuous data distribution $p(x)$ over high-dimensional observations $x \in \mathbb{R}^D$. Standard autoencoders compress inputs into deterministic coordinate vectors $z = f_\theta(x)$ inside a low-dimensional latent space $\mathbb{R}^d$. This deterministic mapping creates an uncontrolled latent topology with vast unmapped gaps: sampling an arbitrary latent vector $z \sim \mathcal{N}(0, I)$ and decoding it produces nonsensical outputs because the decoder was never trained on those empty regions.

VAEs solve this by framing generation probabilistically: we introduce continuous latent variables $z$ governed by a prior distribution $p(z) = \mathcal{N}(0, I)$. The marginal data likelihood is:

$$
p(x) = \int p(x, z)\,dz = \int p_\theta(x|z)\,p(z)\,dz
$$

Direct optimization via Maximum Likelihood Estimation is computationally intractable because the integral requires evaluating all possible latent configurations across high-dimensional space. Furthermore, the true posterior:

$$
p(z|x) = \frac{p_\theta(x|z)\,p(z)}{p(x)}
$$

cannot be computed analytically due to the intractable denominator $p(x)$.

## 2. Variational Inference & The Evidence Lower Bound (ELBO)
To circumvent the intractable true posterior, VAEs introduce an inference network (encoder) $q_\phi(z|x)$ that outputs parameters of a tractable Gaussian distribution:

$$
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \Sigma_\phi(x))
$$

where $\Sigma_\phi(x) = \operatorname{diag}(\sigma_\phi^2(x))$.

We measure the discrepancy between the approximation $q_\phi(z|x)$ and the true posterior $p(z|x)$ using Kullback-Leibler (KL) divergence:

$$
D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z|x)) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p(z|x)}\right]
$$

Expanding this divergence via Bayes' theorem yields:

$$
\log p(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z)) + D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z|x))
$$

Because KL divergence is strictly non-negative ($D_{\mathrm{KL}} \ge 0$), we obtain the Evidence Lower Bound:

$$
\mathcal{L}_{\mathrm{ELBO}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z)) \le \log p(x)
$$

The objective decomposes into two opposing forces:
1. **Reconstruction Log-Likelihood**: $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ penalizes reconstruction errors, driving the latent space to retain discriminative features.
2. **Prior Regularization**: $D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z))$ forces every conditional posterior $q_\phi(z|x)$ to conform to the standard normal prior $\mathcal{N}(0, I)$, preventing isolated clustering and ensuring smooth interpolation across the entire manifold.

## 3. The Reparameterization Trick
Backpropagating gradients through the stochastic sampling step $z \sim q_\phi(z|x)$ is impossible with standard automatic differentiation because the node is stochastic. VAEs resolve this by isolating randomness into an independent auxiliary noise variable:

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$

Because the transformation is deterministic and differentiable with respect to $\mu_\phi$ and $\sigma_\phi$, analytical gradients $\nabla_\phi \mathcal{L}_{\mathrm{ELBO}}$ flow end-to-end through the encoder network.

## 4. Analytical KL Divergence for Gaussians
When both the variational posterior $q_\phi(z|x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ and the prior $p(z) = \mathcal{N}(0, I)$ are Gaussian, the KL divergence resolves to closed form:

$$
D_{\mathrm{KL}}(q_\phi(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^d \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)
$$

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
        content=r"""# Bayes' Theorem & Bayesian Inversion

## 1. Core Principle & Mathematical Statement
Bayes' Theorem is the fundamental law of belief updating under observed evidence. For an unobserved parameter or latent state $z$ and an observed dataset $x$:

$$
P(z|x) = \frac{P(x|z)\,P(z)}{P(x)}
$$

The components represent distinct probabilistic entities:
- **Prior $P(z)$**: The initial probability distribution over hypotheses before observing evidence.
- **Likelihood $P(x|z)$**: The probability of the observed data conditioned on a specific hypothesis $z$.
- **Evidence / Marginal Likelihood $P(x)$**: The total probability of the data averaged over all possible hypotheses:
  

$$
P(x) = \int P(x|z)\,P(z)\,dz
$$

- **Posterior $P(z|x)$**: The updated distribution over hypotheses after incorporating the data $x$.

## 2. The Intractability of the Denominator
In low-dimensional discrete settings, computing $P(x) = \sum_i P(x|z_i)\,P(z_i)$ is trivial. However, when $z \in \mathbb{R}^d$ is high-dimensional (e.g., latent vectors in deep learning, weight spaces in neural networks), the integral:

$$
P(x) = \int_{\mathbb{R}^d} P(x|z)\,P(z)\,dz
$$

has no closed-form solution for complex non-linear models. This intractability directly necessitates approximate inference methods such as Markov Chain Monte Carlo (MCMC) and Variational Inference.

## 3. Base Rate Fallacy & Concrete Numerical Behavior
When the prior probability $P(z)$ of an event is extremely low, even a highly accurate diagnostic test yields a low posterior probability upon a positive test result.
If a disease has prevalence $P(D) = 0.001$, and a test has $99\%$ sensitivity ($P(+|D) = 0.99$) and $95\%$ specificity ($P(-|\neg D) = 0.95$):

$$
P(D|+) = \frac{P(+|D)\,P(D)}{P(+|D)\,P(D) + P(+|\neg D)\,P(\neg D)} = \frac{0.99 \times 0.001}{(0.99 \times 0.001) + (0.05 \times 0.999)} \approx 0.0194
$$

Even after testing positive, the posterior probability is less than $2\%$, because the false-positive volume from the large healthy population overwhelms the true positives.
""",
        intrinsic_score=1.0,
    )

    kl = ConceptBlueprint(
        id="kl-divergence",
        title="KL Divergence",
        topic="Information Theory",
        content=r"""# Kullback-Leibler (KL) Divergence

## 1. Definition & Information-Theoretic Foundations
Kullback-Leibler divergence (relative entropy) measures the statistical discrepancy and coding inefficiency when a true probability distribution $P$ is approximated by distribution $Q$.
For continuous random variables with densities $p(x)$ and $q(x)$:

$$
D_{\mathrm{KL}}(P \parallel Q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)}\,dx = \mathbb{E}_{x \sim P}\left[\log p(x) - \log q(x)\right]
$$

In coding theory, if events occur according to $P$, the optimal code length for $x$ is $-\log_2 p(x)$ bits. Constructing a code assuming $Q$ yields expected excess code length per symbol of exactly $D_{\mathrm{KL}}(P \parallel Q)$.

## 2. Invariants & Asymmetry
1. **Non-negativity ($D_{\mathrm{KL}} \ge 0$)**: Proven via Jensen's inequality applied to the convex function $f(t) = t \log t$:
   

$$
D_{\mathrm{KL}}(P \parallel Q) = \mathbb{E}_{Q}\left[\frac{p(x)}{q(x)} \log \frac{p(x)}{q(x)}\right] \ge f\left(\mathbb{E}_Q\left[\frac{p(x)}{q(x)}\right]\right) = f(1) = 0
$$

   Equality holds if and only if $P = Q$ almost everywhere.
2. **Asymmetry ($D_{\mathrm{KL}}(P \parallel Q) \ne D_{\mathrm{KL}}(Q \parallel P)$)**: KL divergence is not a mathematical metric because it violates symmetry and the triangle inequality.

## 3. Forward vs Reverse KL Optimization
- **Forward KL ($D_{\mathrm{KL}}(P \parallel Q)$)**: Expectation is under $P$. Wherever $p(x) > 0$, $q(x)$ must stay non-zero to avoid $p(x)/q(x) \to \infty$. This produces **zero-avoiding / mean-seeking** behavior: $Q$ stretches to cover all modes of $P$.
- **Reverse KL ($D_{\mathrm{KL}}(Q \parallel P)$)**: Expectation is under $Q$. Wherever $p(x) = 0$, $q(x)$ must collapse to $0$ to avoid infinite penalty. This produces **zero-forcing / mode-seeking** behavior: $Q$ locks onto a single dominant mode of $P$. Variational inference optimizes reverse KL.
""",
        intrinsic_score=1.0,
    )

    bottleneck = ConceptBlueprint(
        id="latent-space-bottleneck",
        title="Latent Space Bottleneck",
        topic="Representation Learning",
        content=r"""# Latent Space Bottleneck

## 1. Information Bottleneck Principle
A latent space bottleneck is an intentional dimensional constriction in a neural architecture that prevents trivial identity mappings and forces representation abstraction.
Given input $X$ and target $Y$, the Information Bottleneck objective maximizes mutual information with the target while minimizing mutual information with the input:

$$
\max_{\theta}  I(Z; Y) - \beta\,I(Z; X)
$$

where $Z = f_\theta(X)$ is the bottleneck representation.

## 2. Dimensionality vs Manifold Discovery
Real-world continuous data (e.g., natural images in $\mathbb{R}^{D}$ with $D = 10^5$) concentrates near a low-dimensional non-linear sub-manifold $\mathcal{M} \subset \mathbb{R}^D$ of intrinsic dimension $d \ll D$.
Constricting the bottleneck dimension to $d$:
1. Filters high-frequency stochastic sensor noise.
2. Disentangles independent generative coordinates.
3. Compresses redundant correlations into compact latent axes.
""",
        intrinsic_score=1.0,
    )

    engine.add_bundle(vae, [bayes, kl, bottleneck])

    bnn = ConceptBlueprint(
        id="bayesian-neural-networks",
        title="Bayesian Neural Networks (BNNs)",
        topic="Machine Learning",
        content=r"""# Bayesian Neural Networks (BNNs)

## 1. Deterministic vs Probabilistic Weight Spaces
Standard neural networks optimize point estimates of parameter matrices:

$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{i} \mathcal{L}(f(x_i; \mathbf{w}), y_i)
$$

This yields overconfident predictions on out-of-distribution inputs because a single parameter vector produces confident outputs even far from training support.

BNNs place a prior distribution over weights $p(\mathbf{w}) = \mathcal{N}(0, \sigma_0^2 I)$. Given dataset $\mathcal{D} = \{(x_i, y_i)\}$, learning computes the weight posterior via Bayes' rule:

$$
p(\mathbf{w}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{w})\,p(\mathbf{w})}{\int p(\mathcal{D}|\mathbf{w})\,p(\mathbf{w})\,d\mathbf{w}}
$$

## 2. Bayes by Backprop (Variational BNNs)
Because integrating over millions of weights is intractable, we fit a variational distribution $q_\theta(\mathbf{w})$ parameterized by mean $\mu$ and unconstrained scale $\rho$ (where $\sigma = \operatorname{softplus}(\rho)$).
Using the reparameterization trick:

$$
\mathbf{w} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The network minimizes the variational free energy (negative ELBO):

$$
\mathcal{F}(\theta, \mathcal{D}) = D_{\mathrm{KL}}(q_\theta(\mathbf{w}) \parallel p(\mathbf{w})) - \mathbb{E}_{q_\theta(\mathbf{w})}[\log p(\mathcal{D}|\mathbf{w})]
$$

## 3. Epistemic vs Aleatoric Uncertainty
- **Aleatoric Uncertainty**: Inherent observational noise in the data-generating process (captured by output distribution variance $\sigma_y^2(x)$).
- **Epistemic Uncertainty**: Parameter uncertainty due to limited data (captured by predictive variance across sampled weights $\mathbf{w}^{(s)} \sim q_\theta(\mathbf{w})$).
""",
        intrinsic_score=1.0,
    )
    engine.add_concept(bnn)

    engine.connect("variational-autoencoders", "bayesian-neural-networks", strength=0.85)
    engine.connect("bayes-theorem", "bayesian-neural-networks", strength=0.90)

    engine.review_joint("bayes-theorem", "variational-autoencoders", score=0.40)

    save_graph(engine, args.graph)
    print(
        f"Initialized Nexus graph at {args.graph} with {len(engine.concepts)} concepts "
        f"and {len(engine.connections)} connections."
    )


def cmd_status(args) -> None:
    """Display overall network integrity, keystones, faults, decay, and isolated concepts."""
    engine = _get_engine(args.graph)
    print(json.dumps(engine.summary(), indent=2, ensure_ascii=False))


def cmd_study(args) -> None:
    """Surface the highest-priority joints and frontier leaves using dual urgency."""
    engine = _get_engine(args.graph)
    if not engine.concepts:
        print("Graph is empty. Run 'python -m nexus.cli init' to initialize.")
        return

    candidates = engine.get_study_candidates(top_k=args.top_k)

    if args.brief:
        briefs = [SocraticTutor.build_study_brief(c) for c in candidates]
        print("\n\n---\n\n".join(briefs))
        return

    output = []
    for c in candidates:
        weak_conn = None
        if c.frayed_connection and c.frayed_target:
            weak_conn = {
                "target_id": c.frayed_connection.target_id,
                "target_title": c.frayed_target.title,
                "target_content": c.frayed_target.content,
                "strength": c.frayed_connection.strength,
                "effective_strength": c.effective_strength,
                "half_life_days": c.frayed_connection.half_life,
                "days_since_review": round(engine.connection_days_elapsed(c.frayed_connection), 2),
            }

        item = {
            "concept_id": c.concept.id,
            "title": c.concept.title,
            "topic": c.concept.topic,
            "review_reason": c.review_reason,
            "urgency_score": c.urgency_score,
            "fault_score": c.fault_score,
            "decay_leverage": c.decay_leverage,
            "stability_score": c.stability_score,
            "effective_strength": c.effective_strength,
            "structural_load": c.structural_load,
            "weak_upward_connection": weak_conn,
            "downstream_impact": [{"id": d.id, "title": d.title} for d in c.downstream_impact],
            "implicit_refresh_targets": c.implicit_refresh_targets,
            "prerequisites": [
                {
                    "id": ing.id,
                    "title": ing.title,
                    "effective_strength": engine.effective_connection_strength(
                        engine.get_connection(ing.id, c.concept.id)
                    ),
                }
                for ing in c.foundational_ingredients
                if engine.get_connection(ing.id, c.concept.id) is not None
            ],
            "target_prerequisites": c.target_prerequisites,
            "content": c.concept.content,
        }
        output.append(item)

    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_diagnose(args) -> None:
    """Rank every supporting joint below a concept to isolate faults top-down."""
    engine = _get_engine(args.graph)
    try:
        diag = engine.diagnose_concept(args.concept_id)
    except KeyError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

    if args.brief:
        print(SocraticTutor.build_diagnostic_brief(diag))
        return

    print(json.dumps(diag, indent=2, ensure_ascii=False))


def cmd_inspect(args) -> None:
    """Print the complete cognitive blueprint, live metrics, and bottleneck radar."""
    engine = _get_engine(args.graph)
    concept = engine.concepts.get(args.concept_id)
    if not concept:
        print(f"Error: Concept {args.concept_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    outgoing = engine.get_outgoing(args.concept_id)
    incoming = engine.get_incoming(args.concept_id)
    stability = engine.calculate_concept_stability(args.concept_id)
    layers = engine.topological_layers()
    downstream = engine.get_downstream_concepts(args.concept_id)
    bottleneck = engine.find_weakest_supporting_joint(args.concept_id)

    res = {
        "blueprint": concept.model_dump(),
        "layer": layers.get(args.concept_id, 0),
        "stability": stability,
        "structural_load": len(downstream),
        "downstream_concepts": [{"id": d.id, "title": d.title} for d in downstream],
        "points_up_into": [
            {
                "target_id": conn.target_id,
                "target_title": engine.concepts[conn.target_id].title if conn.target_id in engine.concepts else "",
                "connection_strength": conn.strength,
                "effective_strength": engine.effective_connection_strength(conn),
                "half_life_days": conn.half_life,
                "review_count": conn.review_count,
                "last_reviewed": conn.last_reviewed,
            }
            for conn in outgoing
        ],
        "built_from_below_by": [
            {
                "source_id": conn.source_id,
                "source_title": engine.concepts[conn.source_id].title if conn.source_id in engine.concepts else "",
                "connection_strength": conn.strength,
                "effective_strength": engine.effective_connection_strength(conn),
                "half_life_days": conn.half_life,
                "review_count": conn.review_count,
                "last_reviewed": conn.last_reviewed,
            }
            for conn in incoming
        ],
        "weakest_supporting_joint": (
            {
                "source_id": bottleneck[0].source_id,
                "target_id": bottleneck[0].target_id,
                "effective_strength": bottleneck[1],
                "hops_below": bottleneck[2],
            }
            if bottleneck
            else None
        ),
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_list(args) -> None:
    """List or search concepts in the graph (lightweight catalog for agents and users)."""
    engine = _get_engine(args.graph)
    matches = engine.search_concepts(query=args.query, topic=args.topic)
    layers = engine.topological_layers()

    catalog = []
    for c in sorted(matches, key=lambda item: (layers.get(item.id, 0), item.id)):
        metrics = engine.calculate_concept_stability(c.id)
        catalog.append({
            "id": c.id,
            "title": c.title,
            "topic": c.topic,
            "layer": layers.get(c.id, 0),
            "stability": metrics["stability"],
            "outgoing_count": metrics["breadth_k"],
            "incoming_count": len(engine.get_incoming(c.id)),
        })
    print(json.dumps(catalog, indent=2, ensure_ascii=False))


def cmd_review(args) -> None:
    """Update health on a directed joint, a frontier concept, or a target's inbound bundle."""
    engine = _get_engine(args.graph)
    propagate = not args.no_propagate

    try:
        if args.scores is not None:
            if not args.target:
                raise ValueError("--scores requires --target <concept_id>")
            scores = json.loads(args.scores)
            result = engine.review_bundle(args.target, scores, propagate=propagate)
            save_graph(engine, args.graph)
            print(json.dumps({"status": "updated", **result}, indent=2, ensure_ascii=False))
            return

        if args.concept:
            if args.score is None:
                raise ValueError("--concept requires --score <0.0..1.0>")
            res = engine.review_concept(args.concept, score=args.score, propagate=propagate)
            save_graph(engine, args.graph)
            print(json.dumps({"status": "updated", **res.model_dump()}, indent=2, ensure_ascii=False))
            return

        if args.source and args.target:
            if args.score is None:
                raise ValueError("--source and --target require --score <0.0..1.0>")
            res = engine.review_joint(args.source, args.target, score=args.score, propagate=propagate)
            save_graph(engine, args.graph)
            print(json.dumps({"status": "updated", **res.model_dump()}, indent=2, ensure_ascii=False))
            return

        raise ValueError(
            "Provide either (--source, --target, --score), (--concept, --score), or (--target, --scores JSON)."
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_connect(args) -> None:
    """Wire a directed upward connection between two existing concepts."""
    engine = _get_engine(args.graph)
    try:
        conn = engine.connect(args.source, args.target, strength=args.strength)
        save_graph(engine, args.graph)
        print(json.dumps({
            "status": "connected",
            "source_id": conn.source_id,
            "target_id": conn.target_id,
            "strength": conn.strength,
        }, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_remove(args) -> None:
    """Remove either a single directed connection or a concept (and its incident edges)."""
    engine = _get_engine(args.graph)
    try:
        if args.source and args.target:
            conn = engine.disconnect(args.source, args.target)
            save_graph(engine, args.graph)
            print(json.dumps({
                "status": "disconnected",
                "source_id": conn.source_id,
                "target_id": conn.target_id,
            }, indent=2))
            return

        if args.concept_id:
            removed = engine.remove_concept(args.concept_id)
            save_graph(engine, args.graph)
            print(json.dumps({
                "status": "removed",
                "concept_id": removed.id,
                "title": removed.title,
            }, indent=2))
            return

        raise ValueError("Provide either a <concept_id> or both --source and --target.")
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_add_concept(args) -> None:
    """Add or update a single concept blueprint."""
    engine = _get_engine(args.graph)
    try:
        if args.data:
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

        if args.file:
            content = _read_text_arg(args.file)
        elif args.content:
            content = args.content
        else:
            raise ValueError("Must provide either --content, --file (use '-' for stdin), or --data JSON.")

        if not args.title:
            raise ValueError("Must specify --title.")

        cid = args.id or slugify(args.title)
        bp = ConceptBlueprint(
            id=cid,
            title=args.title,
            topic=args.topic or "General",
            content=content,
            intrinsic_score=float(args.intrinsic_score),
        )
        engine.add_concept(bp)
        save_graph(engine, args.graph)
        print(json.dumps({
            "status": "added",
            "concept_id": bp.id,
            "title": bp.title,
            "topic": bp.topic,
        }, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_add_bundle(args) -> None:
    """Crystallize a concept and wire its prerequisites pointing UP to it."""
    engine = _get_engine(args.graph)
    try:
        if args.file:
            data = json.loads(_read_text_arg(args.file))
        elif args.data:
            data = json.loads(args.data)
        else:
            raise ValueError("Must provide either --data JSON string or --file path (use '-' for stdin).")

        c_data = data.get("concept") or data.get("target")
        if not c_data:
            raise ValueError("JSON must include a 'concept' object.")

        c_id = c_data.get("id") or slugify(c_data["title"])
        main_bp = ConceptBlueprint(
            id=c_id,
            title=c_data["title"],
            topic=c_data.get("topic", "General"),
            content=c_data["content"],
            intrinsic_score=float(c_data.get("intrinsic_score", 1.0)),
        )

        raw_prereqs = data.get("prerequisites") or data.get("checkpoints") or []
        res = engine.add_bundle(main_bp, raw_prereqs)
        save_graph(engine, args.graph)

        print(json.dumps({
            "status": "bundle_created",
            "concept_id": res["concept"].id,
            "title": res["concept"].title,
            "prerequisites": [{"id": p.id, "title": p.title} for p in res["prerequisites"]],
            "prerequisites_created": [p.title for p in res["prerequisites"]],
            "connections_wired": [
                {"from": c.source_id, "to": c.target_id, "strength": c.strength}
                for c in res["connections"]
            ],
        }, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def cmd_export(args) -> None:
    """Export all concepts as linked standalone Markdown files plus INDEX.md."""
    engine = _get_engine(args.graph)
    res = export_markdown(engine, args.output_dir)
    print(json.dumps({"status": "exported", **res}, indent=2, ensure_ascii=False))


def cmd_graph(args) -> None:
    """Print an ASCII visualization ordered by topological layer (Foundations -> Frontier)."""
    engine = _get_engine(args.graph)
    if not engine.concepts:
        print("Empty graph.")
        return

    layers = engine.topological_layers()
    max_layer = max(layers.values()) if layers else 0
    by_layer: dict[int, list[str]] = {}
    for cid, layer in layers.items():
        by_layer.setdefault(layer, []).append(cid)

    print("================================================================================")
    print("                     NEXUS DIRECTED COGNITIVE NETWORK                           ")
    print("================================================================================")

    for layer_idx in range(max_layer + 1):
        cids = sorted(by_layer.get(layer_idx, []))
        if not cids:
            continue
        if layer_idx == 0:
            label = "LAYER 0: FOUNDATIONS"
        elif layer_idx == max_layer:
            label = f"LAYER {layer_idx}: FRONTIER"
        else:
            label = f"LAYER {layer_idx}"
        print(f"\n--- {label} ---")

        for cid in cids:
            c = engine.concepts[cid]
            outgoing = engine.get_outgoing(cid)
            incoming = engine.get_incoming(cid)
            metrics = engine.calculate_concept_stability(cid)

            print(f"\n[NODE] {c.title} ({c.topic}) [{cid}]")
            print(f"       Stability: {metrics['stability']:.2f} | Outgoing Contexts: {metrics['breadth_k']}")
            if outgoing:
                print("       POINTS UP INTO:")
                for out in outgoing:
                    tgt_title = engine.concepts[out.target_id].title if out.target_id in engine.concepts else out.target_id
                    eff = engine.effective_connection_strength(out)
                    filled = int(round(eff * 10))
                    bar = "█" * filled + "░" * (10 - filled)
                    decay_note = f" (rec {out.strength:.2f})" if abs(eff - out.strength) >= 0.02 else ""
                    print(f"         └───[{bar} {eff:.2f}{decay_note} | h={out.half_life:.1f}d]──► {tgt_title}")
            else:
                eff_leaf = engine.effective_concept_intrinsic(c)
                print(f"       └── (Frontier Leaf | live mastery: {eff_leaf:.2f} | h={c.half_life:.1f}d)")

            if incoming:
                print("       BUILT FROM BELOW BY:")
                for inc in incoming:
                    src_title = engine.concepts[inc.source_id].title if inc.source_id in engine.concepts else inc.source_id
                    eff = engine.effective_connection_strength(inc)
                    print(f"         ├── {src_title} (Live Joint: {eff:.2f})")

    print("\n================================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine"
    )
    parser.add_argument("--graph", default=DEFAULT_GRAPH_FILE, help="Path to graph JSON file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Initialize or pre-seed knowledge graph")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing graph")
    p_init.set_defaults(func=cmd_init)

    p_status = subparsers.add_parser("status", help="Show network integrity, keystones, faults, and decay")
    p_status.set_defaults(func=cmd_status)

    p_study = subparsers.add_parser("study", help="Surface highest-priority joints and frontier leaves")
    p_study.add_argument("--top-k", type=int, default=3, help="Number of candidates to surface")
    p_study.add_argument("--brief", action="store_true", help="Render Markdown Socratic study briefs")
    p_study.set_defaults(func=cmd_study)

    p_diag = subparsers.add_parser("diagnose", help="Rank a concept's supporting cone for fault isolation")
    p_diag.add_argument("concept_id", help="Concept identifier slug")
    p_diag.add_argument("--brief", action="store_true", help="Render Markdown diagnostic brief")
    p_diag.set_defaults(func=cmd_diagnose)

    p_inspect = subparsers.add_parser("inspect", help="Inspect a concept blueprint and its connections")
    p_inspect.add_argument("concept_id", help="Concept identifier slug")
    p_inspect.set_defaults(func=cmd_inspect)

    p_list = subparsers.add_parser("list", help="List or search concepts in the graph")
    p_list.add_argument("--topic", help="Filter by topic substring")
    p_list.add_argument("--query", help="Search across id, title, topic, and content")
    p_list.set_defaults(func=cmd_list)

    p_review = subparsers.add_parser("review", help="Update health on a joint, frontier concept, or bundle")
    p_review.add_argument("--source", help="Prerequisite concept slug (for joint review)")
    p_review.add_argument("--target", help="Higher-level concept slug (for joint or bundle review)")
    p_review.add_argument("--concept", help="Concept slug (for frontier leaf intrinsic review)")
    p_review.add_argument("--score", type=float, help="Health score in [0.0, 1.0]")
    p_review.add_argument("--scores", help="JSON map of {source_id: score} for multi-joint bundle diagnosis")
    p_review.add_argument("--no-propagate", action="store_true", help="Disable asymmetric downward confirmation")
    p_review.set_defaults(func=cmd_review)

    p_connect = subparsers.add_parser("connect", help="Wire an upward connection between two existing concepts")
    p_connect.add_argument("--source", required=True, help="Prerequisite concept slug")
    p_connect.add_argument("--target", required=True, help="Higher-level application slug")
    p_connect.add_argument("--strength", type=float, default=1.0, help="Initial strength in [0.0, 1.0]")
    p_connect.set_defaults(func=cmd_connect)

    p_remove = subparsers.add_parser("remove", help="Remove a concept or a single connection")
    p_remove.add_argument("concept_id", nargs="?", help="Concept slug to remove")
    p_remove.add_argument("--source", help="Prerequisite slug (disconnect edge only)")
    p_remove.add_argument("--target", help="Target slug (disconnect edge only)")
    p_remove.set_defaults(func=cmd_remove)

    p_add_c = subparsers.add_parser("add-concept", help="Add or update a single concept blueprint")
    p_add_c.add_argument("--data", help="JSON string of concept data")
    p_add_c.add_argument("--file", help="Path to Markdown file (or '-' for stdin)")
    p_add_c.add_argument("--content", help="Full Markdown explainer text")
    p_add_c.add_argument("--id", help="Optional concept slug ID")
    p_add_c.add_argument("--title", help="Concept title")
    p_add_c.add_argument("--topic", help="Domain or topic")
    p_add_c.add_argument("--intrinsic-score", type=float, default=1.0, help="Intrinsic score [0.0, 1.0]")
    p_add_c.set_defaults(func=cmd_add_concept)

    p_bundle = subparsers.add_parser("add-bundle", help="Crystallize concept and wire prerequisites pointing UP")
    p_bundle.add_argument("--data", help="JSON string representing the bundle")
    p_bundle.add_argument("--file", help="Path to JSON file (or '-' for stdin)")
    p_bundle.set_defaults(func=cmd_add_bundle)

    p_export = subparsers.add_parser("export", help="Export concepts as linked standalone Markdown files")
    p_export.add_argument("output_dir", help="Destination directory for Markdown wiki")
    p_export.set_defaults(func=cmd_export)

    p_graph = subparsers.add_parser("graph", help="Print topological ASCII overview of the network")
    p_graph.set_defaults(func=cmd_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()