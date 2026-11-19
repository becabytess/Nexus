# Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine

> **A graph-native cognitive architecture where concepts are standalone Markdown explainer documents, connections point UPWARD into higher applications, memory decays continuously with structural anchoring, and mastery propagates asymmetrically downward.**

---

## 1. Motivation: Beyond Flat Flashcards

Traditional spaced repetition systems (Anki, SM-2, FSRS) treat knowledge as disconnected, atomic flashcards:
1. **Isolated Evaluation**: Recalling Bayes' Theorem in isolation does not prove you know how it operates inside a Variational Autoencoder's ELBO.
2. **Symmetric Blame**: Failing a complex topic weakens the whole node without diagnosing *which* mathematical prerequisite broke.
3. **Redundant Review**: Flat schedulers nag you to review basic foundations even when you actively exercise them inside advanced derivations.

**Nexus replaces flat cards with a directed acyclic cognitive dependency graph:**
- **Concepts as Standalone Markdown Documents**: Every node—foundational or advanced—shares one uniform schema (`id`, `title`, `topic`, `content`) containing a complete studyable Markdown guide with LaTeX derivations, invariants, and edge cases.
- **Directed Upward Connections**: Edges strictly point upward from prerequisites into higher applications:
  $$[\text{Prerequisite}] \xrightarrow{\quad w \in [0.0, 1.0],\; h \text{ days}\quad} [\text{Higher Application}]$$

---

## 2. Core Engine Rules (Minimal & Threshold-Free)

Everything in Nexus flows from one shared concave curve, $B(n) = 1 + 0.5 \ln(1 + n)$, and four continuous rules:

### 2.1 Breadth-Anchored Decay & Stability
A concept applied across $K$ higher contexts is structurally anchored and decays more slowly:
$$h_{\text{eff}}(u \to v) = h_{u \to v} \cdot B(K_u), \qquad R = 2^{-\Delta t / h_{\text{eff}}}, \qquad w_{\text{eff}} = w \cdot R$$
Live concept stability combines average live outgoing quality with breadth:
$$\text{Stability}(u) = \bar{w}_{\text{eff}} \cdot B(K_u)$$
*(Frontier leaves with $K_u = 0$ fall back to their live intrinsic mastery.)*

### 2.2 Cram-Proof Spacing Update
On review with continuous score $s \in [0, 1]$, let $\delta = 2s - 1 \in [-1, 1]$:
- **Passing ($\delta \ge 0$)**: $h' = h \cdot 2^{2\delta(1 - R)}$. Cramming immediately ($R \approx 1$) gives zero free inflation; spaced recall at half-life ($R = 0.5$) doubles $h$; delayed recall yields a desirable-difficulty boost.
- **Struggling ($\delta < 0$)**: $h' = \max(0.25,\; h \cdot 2^{\delta})$, contracting half-life immediately.

### 2.3 Unified Dual Urgency (Repair + Leverage)
Splitting the deficit $1 - w_{\text{eff}} = (1 - w) + w(1 - R)$ gives one threshold-free priority rule that is the exact mathematical dual of `review()`:
$$\text{Urgency}(x) = \underbrace{(1 - w(x)) \cdot B(\text{load}(x))}_{\text{Surgical Repair (scales UPWARD)}} \;+\; \underbrace{\text{decay}(x) + \sum_{e \in \text{upstream}(x)} 0.5^d \cdot \text{decay}(e)}_{\text{Implicit Confirmation Leverage (sums DOWNWARD)}}$$
- **Known broken foundations** trigger **bottom-up surgical repair**, weighted by the full downstream tower at risk.
- **Mastered branches decaying over time** trigger **top-down high-leverage checks** at the frontier—pass once to refresh the entire prerequisite cone, or drop into fault isolation if you struggle.

### 2.4 Asymmetric Fault Isolation & Implicit Confirmation
- **Mastery flows downward**: Scoring well on a higher joint or frontier leaf sends damped fractional confirmation ($0.5^d$) down to supporting prerequisites whenever $s > w_{\text{eff}}$.
- **Failure stays isolated**: Struggling propagates *nothing* downward. Use `nexus diagnose` to isolate and weaken only the broken link(s).

---

## 3. Quickstart & CLI Reference

### Install
```bash
python -m pip install -e ".[dev]"
```
*(After installing, run either `nexus ...` or `python -m nexus.cli ...`.)*

### Commands
```bash
# Seed realistic deep-learning demo graph with intact LaTeX Markdown explainers
nexus init

# Bird's-eye network integrity, keystones, active faults, decay & isolated nodes
nexus status

# Surface highest-priority joint or frontier leaf (JSON with full docs, or --brief Markdown)
nexus study --top-k 1
nexus study --top-k 1 --brief

# Top-down fault-isolation cone ranked by vulnerability
nexus diagnose bayesian-neural-networks --brief

# Inspect a concept's document, layer, live strengths, downstream load & weakest bottleneck
nexus inspect variational-autoencoders

# List / search existing concepts (ideal before crystallizing new bundles)
nexus list
nexus list --query "bayes"

# Review a directed joint, a frontier leaf, or isolate faults across a target's prerequisites
nexus review --source bayes-theorem --target variational-autoencoders --score 0.95
nexus review --concept bayesian-neural-networks --score 0.90
nexus review --target variational-autoencoders --scores '{"bayes-theorem": 0.4, "kl-divergence": 0.95}'

# Crystallize a new concept, reusing existing prerequisite IDs alongside new documents
nexus add-bundle --file bundle.json        # or --file - to read from stdin

# Wire or remove connections / concepts
nexus connect --source kl-divergence --target bayesian-neural-networks --strength 0.9
nexus remove --source kl-divergence --target bayesian-neural-networks

# Export all concepts as linked standalone Markdown files + topological INDEX.md
nexus export ./wiki

# Topological ASCII overview (Foundations -> Frontier)
nexus graph
```

---

## 4. Running Tests

```bash
python -m pytest -v
```