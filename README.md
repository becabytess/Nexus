# Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine

> **A graph-native cognitive architecture where concepts are rich personal mental models, connections point UPWARD into higher applications, and review surfaces frayed wires holding up complex reasoning structures.**

---

## 1. Motivation: Beyond Flat Flashcards

Traditional spaced repetition systems (Anki, SM-2, FSRS) treat knowledge as disconnected, atomic flashcards evaluated by binary recall intervals. Even continuous manifold systems that assign scalar masses to points in an embedding space suffer from two fundamental flaws:
1. **Isolated Evaluation**: Answering a question about Bayes' Theorem in isolation does not prove you understand how Bayes' Theorem operates inside the Evidence Lower Bound (ELBO) of a Variational Autoencoder.
2. **Symmetric Blame**: If you fail to derive a complex framework (e.g. VAE), a flat system weakens the VAE node, yet cannot diagnose *which* underlying mathematical prerequisite broke down.

**Nexus replaces flat cards and scalar masses with a directed cognitive dependency graph:**
- Concepts are stored as **rich personal blueprints** (the student's verified mental model, anchor examples, and overcome pitfalls—not generic textbook definitions).
- Directed edges strictly point **UPWARD** from prerequisites into the higher-level applications that depend on them:
  $$[\text{Prerequisite}] \xrightarrow{\quad\text{strength } w \in [0.0, 1.0]\quad} [\text{Higher Application}]$$
- Understanding is evaluated **in context**: a prerequisite is mastered when its outgoing connections to higher-level concepts are resilient and strong.

---

## 2. Core Mathematical Architecture

### 2.1 Directed Upward Topology
Every connection $e = (u, v)$ with weight $w \in [0, 1]$ represents the functional integration of prerequisite $u$ inside target application $v$.

```
[Variational Autoencoders] ◄───(w = 0.85)───┐
      ▲                                      │
      │ (w = 0.40, Frayed!)                  │
[Bayes' Theorem] ───────────────(w = 0.90)──► [Bayesian Neural Networks]
```

When reviewing Bayes' Theorem, Nexus looks **UPWARD** at its outgoing connections. If its joint into VAE is frayed ($w = 0.40$) while its joint into BNN is sound ($w = 0.90$), the tutor targets the exact failure mode inside VAE.

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
- **Struggling on a high-level application**: The LLM tutor inspects the inbound prerequisites, identifies the specific frayed wire via targeted dialogue, and weakens **only** that connection ($w \leftarrow \text{score}$), preserving undamaged prerequisites.

---

## 3. Quickstart & CLI Reference

### 3.1 Installation
```bash
cd nexus
python -m pip install -e .
```

### 3.2 Initialize Demo Network
Pre-seeds a realistic deep learning network (VAEs, BNNs, Bayes' Theorem, KL Divergence, Latent Space Bottleneck) with an intentional frayed link:
```bash
python -m nexus.cli init
```

### 3.3 Study Selection (Scanning Upward)
Surfaces the most urgent concepts whose upward joints are frayed:
```bash
python -m nexus.cli study --top-k 2
```
Output:
```json
[
  {
    "concept_id": "bayes-theorem",
    "title": "Bayes' Theorem",
    "topic": "Probability",
    "urgency_score": 0.7246,
    "reason": "Frayed connection (0.40) into Variational Autoencoders (VAEs). Context: Formulates the true intractable posterior p(z|x) = p(x|z)p(z)/p(x) that VAE approximates with q(z|x).",
    "frayed_joint": {
      "source": "bayes-theorem",
      "target": "variational-autoencoders",
      "target_title": "Variational Autoencoders (VAEs)",
      "strength": 0.4,
      "context_role": "Formulates the true intractable posterior p(z|x) = p(x|z)p(z)/p(x) that VAE approximates with q(z|x)."
    }
  }
]
```

### 3.4 Inspect a Concept Blueprint
Displays the student's personal mental model, outgoing applications (looking UP), and foundational ingredients (looking DOWN):
```bash
python -m nexus.cli inspect bayes-theorem
```

### 3.5 Review & Repair a Joint
Update the strength of a specific connection following Socratic dialogue:
```bash
python -m nexus.cli review --source bayes-theorem --target variational-autoencoders --score 0.95
```

### 3.6 Spontaneous Bundle Crystallization
Record an "aha!" moment by adding a target concept and its supporting checkpoints in one call:
```bash
python -m nexus.cli add-bundle --data '{
  "target": {
    "title": "Diffusion Models",
    "topic": "Generative Models",
    "mental_model": "Inverting a forward Gaussian noise process via learned score matching."
  },
  "checkpoints": [
    {
      "title": "Langevin Dynamics",
      "mental_model": "Sampling from unnormalized densities using gradient of log-density.",
      "context_role": "Provides the sampling trajectory for reverse diffusion."
    }
  ]
}'
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

All 14 tests cover graph operations, upward candidate selection, stability formulas, persistence roundtripping, Socratic scaffolding, and CLI commands.
