# Nexus: Directed Cognitive Blueprint & Connection-Strength Knowledge Engine

> **A graph-native cognitive architecture where concepts are rich personal mental models, connections point UPWARD into higher applications, and review surfaces weak connections holding up complex reasoning structures.**

---

## 1. Motivation: Beyond Flat Flashcards

Traditional spaced repetition systems (Anki, SM-2, FSRS) treat knowledge as disconnected, atomic flashcards evaluated by binary recall intervals. Even continuous manifold systems that assign scalar masses to points in an embedding space suffer from two fundamental flaws:
1. **Isolated Evaluation**: Answering a question about Bayes' Theorem in isolation does not prove you understand how Bayes' Theorem operates inside the Evidence Lower Bound (ELBO) of a Variational Autoencoder.
2. **Symmetric Blame**: If you fail to derive a complex framework (e.g. VAE), a flat system weakens the VAE node, yet cannot diagnose *which* underlying mathematical prerequisite broke down.

**Nexus replaces flat cards and scalar masses with a directed cognitive dependency graph:**
- Concepts are stored as **rich personal blueprints** (the student's verified mental model, anchor examples, and overcome pitfalls—not generic textbook definitions). **All concepts share the exact same schema; no concept is intrinsically special or different.**
- **Multi-Paragraph Depth (Strictly NO One-Liners)**: Brevity is treated as a defect in Nexus. Every `mental_model` must be an exhaustive, multi-paragraph exposition (typically 2-4 comprehensive paragraphs) detailing the first-principles derivation, structural invariants, and mathematical mechanics that produced the student's personal understanding.
- Directed edges strictly point **UPWARD** from prerequisites into the higher-level applications that depend on them:
  $$[\text{Prerequisite}] \xrightarrow{\quad\text{strength } w \in [0.0, 1.0]\quad} [\text{Higher Application}]$$
- **The connection strength $w$ alone creates the hierarchy.** The LLM understands how concepts relate directly from the blueprints.

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
Pre-seeds a realistic deep learning network (VAEs, BNNs, Bayes' Theorem, KL Divergence, Latent Space Bottleneck) with an intentional weak link:
```bash
python -m nexus.cli init
```

### 3.3 Study Selection (Scanning Upward)
Surfaces the most urgent concepts whose upward joints are weak:
```bash
python -m nexus.cli study --top-k 1
```
Output:
```json
[
  {
    "concept_id": "bayes-theorem",
    "title": "Bayes' Theorem",
    "topic": "Probability",
    "urgency_score": 0.7978,
    "stability_score": 1.0069,
    "weak_upward_connection": {
      "target_id": "variational-autoencoders",
      "target_title": "Variational Autoencoders (VAEs)",
      "strength": 0.4
    },
    "prerequisites": [],
    "mental_model": "Belief re-weighting under evidence...",
    "anchor_example": "Medical testing: high false-positive rates when the prior prevalence is low."
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
Record an "aha!" moment by adding a concept and its supporting prerequisites in one call. **Every concept blueprint must be a serious, multi-paragraph technical breakdown (strictly NO one-liners or brief glosses)**:
```bash
python -m nexus.cli add-bundle --data '{
  "concept": {
    "title": "Diffusion Probabilistic Models",
    "topic": "Generative Models",
    "mental_model": "A generative framework based on a parameterized Markov chain that reverses a gradual noise-injection process. In the forward process q(x_t | x_{t-1}), Gaussian perturbation is iteratively added over T discrete steps according to a variance schedule beta_t, eventually degrading any data distribution p(x_0) into pure isotropic Gaussian noise N(0, I). By reparameterization, any intermediate state x_t can be sampled directly in closed form from x_0 via alpha_bar_t: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.\n\nThe generative model p_theta(x_{t-1} | x_t) learns to reverse this degradation step-by-step. Instead of directly predicting the clean image x_0, the neural network (typically a U-Net conditioned on time step t) is trained with a simplified L2 objective to predict the exact noise vector epsilon added at step t: L_simple = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]. This parameterization is mathematically equivalent to score-matching on smoothed densities: the predicted noise directly yields the score function nabla_{x_t} log p(x_t).\n\nSampling proceeds backward from x_T ~ N(0, I) down to x_0 by iteratively subtracting the network predicted noise and adding scaled Gaussian perturbations (for t > 1). This steady thermodynamic reversal prevents mode collapse common in GANs and avoids the blurry interpolations typical of under-regularized autoencoders.",
    "anchor_example": "Walking through the closed-form forward marginal: showing that as t approaches T, alpha_bar_t approaches 0, reducing the conditional distribution q(x_T | x_0) strictly to N(0, I), completely destroying input signal. In the reverse step, verifying that setting variance sigma_t^2 = 0 turns the stochastic DDPM into a deterministic DDIM ordinary differential equation (ODE) trajectory.",
    "pitfalls": "Confusing DDPM noise prediction with autoencoder denoising. Standard denoising autoencoders map directly to clean data x_0 in a single step, which averages multimodal possibilities into a blurry mean. Diffusion succeeds because it predicts infinitesimal noise increments over an extended trajectory, resolving ambiguities incrementally across distinct frequency scales."
  },
  "prerequisites": [
    {
      "title": "Langevin Dynamics & Score Matching",
      "topic": "Statistical Physics & ML",
      "mental_model": "A Markov Chain Monte Carlo (MCMC) method that generates samples from an unnormalized probability density p(x) using only the gradient of its log-density—termed the score function nabla_x log p(x). The iterative update x_{k+1} = x_k + (epsilon / 2) * nabla_x log p(x_k) + sqrt(epsilon) * z where z ~ N(0, I) combines gradient ascent toward high-density modes with Brownian noise exploration.\n\nThe critical advantage is that the intractable normalizing partition function Z in p(x) = p_tilde(x) / Z vanishes upon taking the spatial gradient: nabla_x log p(x) = nabla_x log p_tilde(x) - nabla_x log Z = nabla_x log p_tilde(x). Score matching trains a model s_theta(x) to approximate this gradient field without ever computing or approximating Z.\n\nIn low-density data regions, score estimation is fundamentally inaccurate because samples rarely appear there during training. Diffusion models resolve this fatal limitation of Langevin dynamics by perturbing the data at multiple noise scales, ensuring non-zero density everywhere and providing a gradient highway connecting noise to data modes.",
      "anchor_example": "Writing out the update for a 2D Gaussian mixture model: demonstrating that without the noise term sqrt(epsilon) * z, the update is pure gradient ascent collapsing to local modes. With the Langevin noise term, the trajectory correctly samples according to the relative mass of each Gaussian component.",
      "pitfalls": "Believing that Langevin dynamics requires knowing the partition function Z. Its entire purpose and power lie in operating strictly on score vectors, rendering normalizing constants completely irrelevant."
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
All 14 tests pass.
