# Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine

> **A graph-native cognitive architecture where concepts are standalone Markdown explainer documents, connections point UPWARD into higher applications, and review surfaces weak connections holding up complex reasoning structures.**

---

## 1. Motivation: Beyond Flat Flashcards

Traditional spaced repetition systems (Anki, SM-2, FSRS) treat knowledge as disconnected, atomic flashcards evaluated by binary recall intervals. Even continuous manifold systems that assign scalar masses to points in an embedding space suffer from two fundamental flaws:
1. **Isolated Evaluation**: Answering a question about Bayes' Theorem in isolation does not prove you understand how Bayes' Theorem operates inside the Evidence Lower Bound (ELBO) of a Variational Autoencoder.
2. **Symmetric Blame**: If you fail to derive a complex framework (e.g. VAE), a flat system weakens the VAE node, yet cannot diagnose *which* underlying mathematical prerequisite broke down.

**Nexus replaces flat cards and scalar masses with a directed cognitive dependency graph:**
- **Concepts as Standalone Markdown Documents**: A concept is not a tag, card, or brief summary. Each concept is a complete, studyable Markdown explainer document complete with mathematical derivations in LaTeX ($$...$$), structural invariants, and edge-case breakdowns that someone can read top-to-bottom to master the concept.
- **Uniform Schema**: Every concept—whether a foundational building block like Bayes' Theorem or a complex framework like VAE—shares the exact same clean schema (`id`, `title`, `topic`, `content`).
- **Directed Upward Connections**: Connections strictly point from foundational prerequisites into the higher-level applications that depend on them:
  $$[\text{Prerequisite}] \xrightarrow{\quad\text{strength } w \in [0.0, 1.0]\quad} [\text{Higher Application}]$$
- **Topology Alone Creates Hierarchy**: The connection strength $w$ defines the structural dependencies. The LLM understands how concepts relate directly from their Markdown content.

---

## 2. Core Architecture

### 2.1 Directed Upward Topology
Every connection $e = (u, v)$ with weight $w \in [0, 1]$ represents the strength of prerequisite $u$ supporting target application $v$.

```
[Variational Autoencoders] ◄───(w = 0.85)───┐
      ▲                                      │
      │ (w = 0.40, Weak!)                    │
[Bayes' Theorem] ───────────────(w = 0.90)──► [Bayesian Neural Networks]
```

When reviewing Bayes' Theorem, Nexus looks **UPWARD** at its outgoing connections. If its joint into VAE is weak ($w = 0.40$) while its joint into BNN is sound ($w = 0.90$), the tutor targets the exact failure mode inside VAE.

### 2.2 Stability Metric: Quality $\times$ Breadth
A concept applied across many distinct domains is far more consolidated than a concept learned in a single chapter. To avoid high volume faking mastery, Nexus computes concept stability as the product of **average connection quality** and a concave **breadth multiplier**:

$$\text{Stability}(u) = \bar{w}_u \times \left(1.0 + 0.5 \ln(1 + K_u)\right)$$

where:
- $K_u = |\text{Out}(u)|$ is the number of higher applications relying on concept $u$.
- $\bar{w}_u = \frac{1}{K_u} \sum_{e \in \text{Out}(u)} w_e$ is the average health of outgoing connections.
- For frontier leaf concepts ($K_u = 0$, top of current horizon), stability falls back to the concept's intrinsic baseline score.

| Outgoing Contexts ($K$) | Breadth Multiplier | Stability ($\bar{w} = 0.90$) | Stability ($\bar{w} = 0.50$) |
|---|---|---|---|
| $0$ (Frontier Leaf) | $1.00$ | $0.90$ (Intrinsic) | $0.50$ (Intrinsic) |
| $1$ Context | $1.35$ | $1.21$ | $0.67$ |
| $3$ Contexts | $1.69$ | $1.52$ | $0.85$ |
| $7$ Contexts | $2.04$ | $1.84$ | $1.02$ |

### 2.3 Asymmetric Socratic Diagnosis
- **Mastering a high-level application** ($A \to B$ correct): Reinforces all inbound prerequisite joints feeding $B$.
- **Struggling on a high-level application**: The LLM tutor inspects the inbound prerequisites, identifies the specific weak link via targeted dialogue, and weakens **only** that connection ($w \leftarrow \text{score}$), preserving undamaged prerequisites.

---

## 3. Quickstart & CLI Reference

### 3.1 Installation
```bash
cd nexus
python -m pip install -e .
```

### 3.2 Initialize Demo Network
Pre-seeds a realistic deep learning network (VAEs, BNNs, Bayes' Theorem, KL Divergence, Latent Space Bottleneck) with full Markdown explainer documents and an intentional weak link:
```bash
python -m nexus.cli init
```

### 3.3 Study Selection (Scanning Upward)
Surfaces the most urgent concepts whose upward joints are weak:
```bash
python -m nexus.cli study --top-k 1
```

### 3.4 Inspect a Concept Blueprint
Displays the concept's Markdown explainer document, outgoing applications (looking UP), and foundational ingredients (looking DOWN):
```bash
python -m nexus.cli inspect bayes-theorem
```

### 3.5 Review & Repair a Joint
Update the strength of a specific connection following Socratic dialogue:
```bash
python -m nexus.cli review --source bayes-theorem --target variational-autoencoders --score 0.95
```

### 3.6 Spontaneous Bundle Crystallization
Record an "aha!" moment by adding a concept and its supporting prerequisites in one call. Both main concept and prerequisites share the identical Markdown document schema:
```bash
python -m nexus.cli add-bundle --data '{
  "concept": {
    "title": "Diffusion Probabilistic Models",
    "topic": "Generative Models",
    "content": "# Diffusion Probabilistic Models\n\n## 1. Forward Noising Process\nq(x_t | x_{t-1}) adds Gaussian perturbation over T steps...\n\n## 2. Reverse Denoising & Score Matching\nPredicted noise directly estimates score function nabla_{x_t} log p(x_t)..."
  },
  "prerequisites": [
    {
      "title": "Langevin Dynamics & Score Matching",
      "topic": "Statistical Physics & ML",
      "content": "# Langevin Dynamics\n\n## 1. Core Formulation\nSampling from unnormalized densities without the partition function Z...\n$$x_{k+1} = x_k + \\frac{\\epsilon}{2} \\nabla_x \\log p(x_k) + \\sqrt{\\epsilon} z$$"
    }
  ]
}'
```

You can also add a concept directly from a local `.md` file:
```bash
python -m nexus.cli add-concept --title "Information Bottleneck" --topic "Information Theory" --file ./notes/bottleneck.md
```

### 3.7 ASCII Graph Visualization
```bash
python -m nexus.cli graph
```

---

## 4. Running Tests

Run the full pytest suite:
```bash
python -m pytest tests -v
```
All 14 tests pass.
