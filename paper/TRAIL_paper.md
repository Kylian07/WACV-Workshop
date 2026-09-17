# TRAIL: Reasoning in Visual Latent Space as Certified Mirror Descent

*Draft skeleton for the 1st Workshop on Latent Visual Reasoning (LVR), WACV 2027.*
Every theorem below is checked numerically by `tests/test_theory.py`; every figure is
produced by `trail/analysis/figures.py`.

---

## Abstract (draft)

Latent visual reasoning iterates a hidden state in a continuous visual embedding space
rather than decoding intermediate tokens. The recursions in use are free-form: a residual
update `z ← z + f(z, c)` repeated a fixed number of times. Nothing constrains the state to
remain in the region of embedding space the visual encoder actually produces, nothing tells
the model when the reasoning is finished, and nothing makes the intermediate states legible.

We show that a single change of parameterisation supplies all three. Represent the
reasoning state as a distribution `p` over the image's own visual atoms and take each
reasoning step to be an entropic mirror-descent step on a question-conditioned energy

    E_h(p) = -⟨s_h, p⟩ + (β/2)‖p − M_h p‖²,

where `s_h` scores atoms against the current sub-question and `M_h` is a column-stochastic
*relational transport* operator. Then (i) the visual state `z = Vᵀp` lies in `conv(V)` at
every step, so off-manifold drift is zero by construction; (ii) the Frank–Wolfe gap
`G = ⟨∇E, p⟩ − min_i ∇E_i` is computable in `O(N)` and upper-bounds the true suboptimality,
giving a *sound* halting certificate in place of a fixed step count or a learned gate; and
(iii) `p` is a distribution over image regions, so the reasoning trace is the computation
rather than a post-hoc explanation of it. We prove the energy is `4β`-smooth relative to the
negative entropy **independently of the number of atoms**, which prescribes the step size
`η = 1/(4β)` and gives an `O(4β log N / T)` rate. Softmax cross-attention is recovered
exactly as one step of the method.

Under matched budgets, TRAIL matches or exceeds free-form latent recursion, MAC and FiLM on
CLEVR while spending [X]× fewer reasoning steps on average; the step at which its
certificate fires correlates with CLEVR's ground-truth program depth (Spearman ρ = [X])
without ever being supervised on it; and on a depth-extrapolation split it recovers [X]
points over fixed-depth recurrences purely by iterating longer at test time.

---

## 1 Introduction

The argument in four moves:

1. **Latent reasoning is the right idea.** Decoding every intermediate step to text is
   wasteful and lossy for visual sub-goals; reasoning in the embedding space is faster and
   keeps visual information visual.
2. **The current instantiation gives up too much.** A free-form residual recursion has no
   invariant, no termination criterion, and no readable state. Depth is a hyperparameter;
   practitioners report degradation past a few latent steps, which is what an unconstrained
   iterated map should be expected to do.
3. **The fix is not a regulariser, it is a parameterisation.** Ask what set the reasoning
   state should live in. The answer the encoder already gives is: the convex hull of the
   image's own atoms. Optimising over that set has a canonical algorithm — entropic mirror
   descent — and that algorithm arrives with a convergence theory and a computable gap.
4. **What that buys is not one number but three properties at once**, plus backward
   compatibility: attention is the `T = 1` case.

**Contributions.**
(C1) A formulation of latent visual reasoning as constrained convex optimisation over the
simplex of visual atoms, with the relational structure of the question entering as a
column-stochastic transport operator.
(C2) Lemma 1: dimension-free relative smoothness `L = 4β`, hence a *prescribed* step size
and an `O(4β log N / T)` rate that does not degrade as the atom grid is refined.
(C3) Theorem A: a sound, `O(N)`-cost halting certificate; and Prop. 3, which predicts that
halting time scales as `1/Δ` in the decision margin — i.e. that harder questions get more
compute, for free.
(C4) Empirically: an accuracy/compute frontier traced by one trained model, halting steps
that track ground-truth program depth without supervision, zero measured drift, and
improved depth extrapolation.

## 2 Related work

* **Latent / continuous chain-of-thought.** Reasoning in hidden space rather than token
  space; latent visual reasoning in MLLMs. Position TRAIL as *how to iterate*, orthogonal to
  *what to iterate on*, and applicable inside those systems.
* **Unrolled optimisation as architecture.** LISTA, deep equilibrium models, mean-field CRF
  as RNN, deep declarative networks. TRAIL differs in that the optimisation variable is the
  reasoning state itself and the stopping rule is the algorithm's own certificate.
* **Adaptive computation.** ACT/PonderNet learn a halting distribution and pay for it with
  an extra loss term and a learned gate. TRAIL's rule is a bound, not a prediction.
* **Compositional visual reasoning.** MAC, FiLM, NS-VQA, neural module networks — fixed
  depth, or depth supplied by a parsed program. TRAIL infers depth from a certificate.
* **Attention as optimisation.** Attention-as-MAP-inference and energy-based attention.
  Prop. 1 places softmax attention as the first iterate of this scheme.

## 3 Method

### 3.1 Setup

An encoder yields atoms `V ∈ R^{N×d}` (L2-normalised; `N = 14×14` grid cells in our CLEVR
setting). A question encoder yields contextual word states `W` and a summary `q`.

**Reasoning state.** `p ∈ Δ^{N−1}`, with visual state `z = Vᵀp`.

### 3.2 The two loops

*Outer* (`h = 1..H`, one per hop): a control unit reads the next sub-question as a convex
combination of word states, `c_h = Σ_l α_l w_l`, and emits a **fixed** energy

    E_h(p) = −⟨s_h, p⟩ + (β/2)‖p − M_h p‖²,     s_h,i = ⟨V_i, W_s c_h⟩/√d,

with `M_h` column-stochastic, built from a control-conditioned attention over atom pairs
plus a control-conditioned relative-position bias (so that "left of" can become an actual
direction of transport on the grid).

*Inner* (`k = 1..K`): entropic mirror descent on that fixed energy,

    p ← softmax(log p − η ∇E_h(p)),     η = 1/(4β).

The energy is held fixed inside the inner loop, which is exactly what makes §4 apply. It
is also cheaper: `M_h` is built once per hop, not once per update.

**Interpretation of the quadratic term.** `‖p − M_h p‖²` asks the belief to be a fixed
point of the queried relation. Attending to *the thing to the left of the red cube* means
attending to a set that maps to itself under "left of ∘ (start at the red cube)". The
relational term is what turns a re-weighting of evidence into a hop.

### 3.3 Halting

Two certificates, no learned gate:

* **inner** — `G_k ≤ ε·G_0`: this hop is solved to relative accuracy ε (Thm. A);
* **outer** — `KL(p_h‖p_{h−1}) ≤ δ`: the hop did not move the belief, so there is no further
  sub-question.

Sweeping ε on a *single trained model* traces an accuracy/compute curve (Fig. 4).

### 3.4 Read-out

`z_T` is scale-free, being a convex combination; counting needs scale. The head therefore
also receives the normalised entropy of `p_T` and a soft cardinality
`n̂ = Σ_i σ(a·s_i + b)`, which is itself interpretable and is plotted against ground-truth
object counts.

### 3.5 Training

Supervise **every prefix**, `L = (1/H) Σ_h CE(head(p_h), y)`. A model whose stopping time is
data-dependent must be correct at whatever time it stops; the ablation `anytime_loss='last'`
shows this is what makes the frontier flat rather than cliffed.

## 4 Theory

Throughout, `M` is column-stochastic, `h(p) = Σ p_i log p_i`, `D(q,p) = KL(q‖p)`.

---

**Lemma 1 (dimension-free relative smoothness).** `H = β(I−M)ᵀ(I−M)` satisfies
`max_{ij}|H_{ij}| ≤ 4β` for every `N`. Hence `E` is `4β`-smooth w.r.t. `‖·‖₁` on `Δ^{N−1}`
and, via Pinsker, satisfies the Bregman descent lemma with modulus `L = 4β`.

*Proof.* `H_{ij} = β⟨(I−M)e_i, (I−M)e_j⟩`. Column `i` of `M` lies in the simplex, so
`‖Me_i‖₂ ≤ ‖Me_i‖₁ = 1` and `‖(I−M)e_i‖₂ ≤ 1 + 1 = 2`. Cauchy–Schwarz gives
`|H_{ij}| ≤ 4β`. For `f` with `‖∇²f‖_{1→∞} ≤ L`,
`f(q) ≤ f(p) + ⟨∇f(p), q−p⟩ + (L/2)‖q−p‖₁²`, and Pinsker's `‖q−p‖₁² ≤ 2D(q,p)` yields
`f(q) ≤ f(p) + ⟨∇f(p), q−p⟩ + L·D(q,p)`. ∎

The constant does not involve `N`. Refining the atom grid does not change the prescribed
step size — which is why the same weights can be run at a different resolution or for more
iterations at test time.

---

**Theorem B (monotone descent and rate).** With `η ≤ 1/L = 1/(4β)`, the inner loop
satisfies `E(p_{k+1}) ≤ E(p_k) − (1/η − L)·D(p_{k+1}, p_k)`; in particular `E` is
non-increasing, and strictly decreasing for `η ≤ 1/(2L)`. From the uniform start,
`E(p_K) − E* ≤ 4β·log N / K`.

*Proof.* Optimality of the mirror step `p_{k+1} = argmin_p ⟨g_k,p⟩ + (1/η)D(p,p_k)` against
the feasible point `p_k` gives `⟨g_k, p_{k+1}−p_k⟩ + (1/η)D(p_{k+1},p_k) ≤ 0`. Add Lemma 1's
descent lemma. The rate is the standard relatively-smooth mirror-descent bound with
`D(p*, uniform) = log N − H(p*) ≤ log N`. ∎

---

**Theorem A (sound halting certificate).** For convex `E` and `g = ∇E(p)`,
`G(p) := ⟨g,p⟩ − min_i g_i ≥ E(p) − min_{u∈Δ} E(u) ≥ 0`. Stopping the first time `G ≤ ε`
therefore returns an ε-optimal belief, whatever the stopping time turned out to be.

*Proof.* Convexity: `E(u) ≥ E(p) + ⟨g, u−p⟩` for all `u ∈ Δ`, so
`E(p) − E(u) ≤ ⟨g, p−u⟩ ≤ max_{u∈Δ}⟨g, p−u⟩ = G(p)`. The maximum of a linear function over
the simplex is at a vertex, giving the stated closed form. Non-negativity: take `u = p`. ∎

Cost: one max and one dot product, `O(N)`. No extra parameters, no auxiliary loss.

---

**Proposition 1 (attention is one step).** With `p₀` uniform, `β = 0`, `η = 1`, `K = 1`:
`p₁ = softmax(s)` and `z₁ = Vᵀ softmax(s)` — exactly softmax cross-attention.

*Proof.* `∇E = −s`; the mirror step gives `p₁ ∝ p₀ ⊙ e^{s} ∝ e^{s}`. ∎

So TRAIL is a strict generalisation of the attention block it replaces, and `attn1` in the
experiments is the control that isolates iteration from architecture.

---

**Proposition 2 (no off-manifold drift).** `z_t = Vᵀp_t ∈ conv(V)`, hence
`dist(z_t, conv(V)) = 0` for all `t`. For a free-form recursion `z_{t+1} = z_t + f(z_t,c_t)`
no such bound exists, and the quantity is measured in Fig. 3.

---

**Proposition 3 (difficulty-adaptive halting).** For `β = 0`, MWU from the uniform belief
gives `p_t = softmax(t·η·s)` in closed form, so with `S = max_i s_i − min_i s_i` and margin
`Δ = s_(1) − s_(2)`:

    G_t ≤ (N−1)·S·e^{−tηΔ},    hence   T(ε) ≤ ⌈log((N−1)S/ε) / (ηΔ)⌉.

Halting time is inversely proportional to the decision margin: **ambiguous inputs are given
more steps, with no supervision on the number of steps.** §5.3 tests the prediction against
CLEVR's ground-truth program depth.

---

**What the theory does *not* claim.** Thm. A certifies optimality *of the hop's energy*, not
correctness of the answer. The bridge from "this hop is solved" to "the answer is right"
rests on the energy being a good proxy, and is an empirical question — which is precisely
what §5.3 measures. Stating this explicitly is cheaper than having a reviewer state it.

## 5 Experiments

**Protocol.** Every model shares the backbone, question encoder, read-out head, optimiser
and budget; only the reasoning module differs. Budgets are matched in *belief updates*
(TRAIL `H×K = 4×3 = 12`; baselines 12, which is also MAC's standard depth). Numbers are not
comparable to published CLEVR results obtained with ResNet-101 conv4 features and days of
training; this is an internal comparison and is described as such.

* **5.1 HOPWORLD** (controlled): exact hop counts, so ρ(halt, hops) and depth extrapolation
  are measurable without confounds. Table 1.
* **5.2 CLEVR** (iid): accuracy vs average steps; the ε-sweep frontier of Fig. 4. Table 2.
* **5.3 Does the certificate find difficulty?** Halting step vs CLEVR program depth
  (Spearman ρ), Fig. 6. The key *unsupervised* result.
* **5.4 Traces** — `p_t` over the atom grid, Fig. 7; soft cardinality vs true object count.
* **5.5 Generalisation** — depth-extrapolation split (train ≤10 program nodes, test ≥15),
  evaluated at `H` and at `2H` steps; held-out attribute combination (red ∧ cube).
* **5.6 Ablations** — β=0, learned η, last-step loss, flat (K=1), fixed T. Table 3.

## 6 Limitations

* The certificate bounds the hop's energy, not answer correctness (§4).
* The control unit is an unconstrained recurrence; only the *visual* state is drift-free.
  Extending the guarantee to the control path is open.
* `M_h` is `O(N²)` per hop; fine at `N = 196`, and it is built once per hop rather than once
  per update, but it is the scaling bottleneck for high-resolution atom grids. Low-rank or
  sparse transport is the obvious next step.
* Convexity of `E_h` holds for the relational penalty as written; other relational terms
  (e.g. a concave coherence reward) would lose Thm. A's global guarantee and leave only
  stationarity.
* Results here are on a from-scratch reasoner over frozen features. Whether the same
  parameterisation helps inside a pretrained MLLM's latent reasoning loop is untested and is
  the natural follow-up.

## 7 Reproducibility

`tests/test_theory.py` verifies all ten formal claims numerically in under a minute.
`notebooks/trail_wacv2027_kaggle.ipynb` runs the whole study on Kaggle.
