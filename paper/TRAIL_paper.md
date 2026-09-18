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

We report what this buys and what it costs. The halting certificate halves the compute at
identical accuracy, drift is zero to four orders of magnitude, and the trace is the
computation rather than a saliency map over it. Against that, a matched comparison -- in
which every baseline is given the same relational prior -- shows the formulation **does not**
improve accuracy: MAC exceeds it by 9.3 points and a single relational attention step, the
T=1 case of our own method, exceeds it by 8.4 while spending one belief update against
twelve. We also report that the certificate's halting step does not track ground-truth
reasoning depth, and that iterating longer never helps. We give mechanistic accounts of each
and argue that our synthetic benchmark, on which one dense transport application suffices,
cannot test iterated reasoning at all.

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
(C3) Theorem A: a sound, `O(N)`-cost halting certificate for the inner loop, paired with a
belief-stationarity rule for the outer loop; and Prop. 3, which predicts that halting time
scales as `1/Δ` in the decision margin — i.e. that harder questions get more compute, for
free.
(C4) Empirically: an accuracy/compute frontier traced by one trained model, zero measured
drift, a decodable trace, and the best out-of-distribution accuracy of any model tested at
matched budget. We also report **two negative results**: the halting step does not track
ground-truth hop count (§5.3), and iterating longer does not improve depth extrapolation for
any model, tied or untied (§5.5).

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

Two rules, each governing its own loop, and neither a learned gate:

* **inner** — `G_k ≤ ε·G_0`: by Thm. A this hop's energy is solved to relative accuracy ε,
  so stop iterating *within* the hop. This is what the reported compute cost counts.
* **outer** — `KL(p_h‖p_{h−1}) ≤ δ·KL(p_1‖p_0)`: the hop moved no belief, so there is no
  further sub-question. This decides *how many hops* to take, and therefore the answer.

Keeping them apart is not a detail. The gap certifies that the *current sub-question* is
answered, which is not the same as the question being answered: a 3-hop question can have
hop 1 solved to machine precision while the answer is still two hops away. Using the gap to
decide the number of hops conflates the two and stops early on exactly the hard questions
adaptivity was supposed to help — we measured a ~16-point accuracy loss at the operating
point before separating them.

Both thresholds are relative to the first value in their own sequence, which makes a single
global threshold meaningful across examples whose score scales differ. Sweeping δ on a
*single trained model* traces the accuracy/compute curve of Fig. 4; every fixed-depth
baseline can only contribute a point.

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
and budget; only the reasoning module differs. Budgets are matched in **belief updates**,
not in "steps": TRAIL's `H×K = 4×3 = 12` updates against 12 iterations for the iterative
baselines, which is also MAC's standard depth. FiLM is not an iterative reasoner — its
"steps" are residual conv blocks — so it keeps its standard depth of 4 and is reported as
such rather than pretending to a matched budget. Reporting hops and iterations on the same
axis would compare two different units, so every table and Fig. 4's x-axis is in belief
updates. Note that TRAIL builds its `O(N²)` transport operator once per *hop*, so at the
same update budget it performs fewer of the expensive operations than a flat recursion.

Numbers are not comparable to published CLEVR results obtained with ResNet-101 conv4
features at full resolution and days of training; this is an internal comparison at matched
budget and is described as such.

* **5.1 HOPWORLD** (controlled): exact hop counts, so ρ(halt, hops) and depth extrapolation
  are measurable without confounds. Table 1.
* **5.2 CLEVR** (iid): accuracy vs average steps; the ε-sweep frontier of Fig. 4. Table 2.
* **5.3 Does the certificate find difficulty?** Halting step vs ground-truth hop count.
  **On HOPWORLD it does not**, and the negative result is reported in §5.3 below rather
  than buried. Fig. 6.
* **5.4 Traces** — `p_t` over the atom grid, Fig. 7; soft cardinality vs true object count.
* **5.5 Generalisation** — held-out attribute combination (red ∧ cube). The
  depth-extrapolation claim was **withdrawn**; see §5.5 below for the measurement
  that killed it.
* **5.6 Ablations** — β=0, learned η, last-step loss, flat (K=1), fixed T. Table 3.

### 5.3 (measured) The halting rule does not track ground-truth difficulty

Prop. 3 predicts that halting time scales as `1/Δ` in the decision margin, so harder
questions should get more steps for free. On HOPWORLD, **it does not happen.** Sweeping the
outer threshold δ on a trained TRAIL model (24k train / 3k val, 4 hops × 3 inner steps):

| δ | mean hops | std | ρ(halt, true hops) | accuracy |
|---|---|---|---|---|
| 0.50 - 0.10 | 2.00 | 0.000 | undefined | 73.3 |
| 0.05 | 2.00 | 0.026 | 0.011 | 73.3 |
| 0.02 | 2.21 | 0.408 | 0.057 | 76.7 |
| 0.01 | 2.64 | 0.603 | 0.076 | 79.3 |
| 0.00 (fixed depth) | 4.00 | 0.000 | undefined | 81.2 |

Two things are worth separating. Over most of the threshold range the rule **saturates**:
every example halts at exactly hop 2, so ρ is undefined for lack of variance rather than
measurably zero. Where there is variance (δ = 0.01-0.02), ρ = 0.06-0.08, i.e. nothing.
Reporting ρ at a single threshold would have hidden which of these was happening, which is
why the code reports the spread alongside the correlation at every threshold.

**Why.** Prop. 3 is a statement about iterating a *fixed* energy, and it is true there (the
test verifies the bound). The gap between theory and measurement is that `M_h` is a dense
learned operator, not a single ground-truth relation: one application can compose several
hops of the underlying task, so the number of TRAIL hops has no reason to equal the number
of program hops. The theory bounds how long it takes to solve *a* hop; it says nothing about
how many of the task's hops the model chooses to pack into one.

This is a concrete, testable diagnosis rather than a shrug, and it points at the obvious
follow-up: constrain `M_h` (low rank, spatial locality, or a penalty on composing) so that
one application is one relation, and re-measure ρ. Until that is done, the paper claims
compute adaptivity and interpretability, **not** unsupervised difficulty estimation.

### 5.5 (measured) Iterating longer does not help, and weight-tying does not rescue it

The intended claim was that TRAIL, handed questions needing more hops than anything in
training, could recover accuracy by running more mirror-descent steps on the same weights.
A first measurement showed no gain. We hypothesised the cause was the per-hop control
embedding: a model with H learned sub-question slots has no way to express the (H+1)-th, so
iterating it longer cannot help, and the experiment would be measuring the embedding table
rather than the method. We therefore re-trained every model with the hops tied to a single
shared operator (`share_steps=True`) and measured both conditions under identical budgets.

**The hypothesis was wrong.** Tying changes essentially nothing, and no model gains from
extra iterations. Train on 1-3 hop HOPWORLD questions, test on 4-6:

| run | iid | OOD @ native | OOD @ 4x | gain from iterating |
|---|---|---|---|---|
| TRAIL, tied | 0.775 | 0.658 | 0.569 | **-0.089** |
| TRAIL, untied | 0.767 | 0.659 | 0.548 | **-0.111** |
| free-form latent, tied | 0.524 | 0.437 | 0.438 | +0.000 |
| free-form latent, untied | 0.534 | 0.438 | 0.442 | +0.003 |
| MAC, tied | 0.610 | 0.494 | 0.420 | -0.074 |
| MAC, untied | 0.608 | 0.503 | 0.464 | -0.039 |

Matched tied-against-untied, the difference is +0.008 (TRAIL), -0.010 (free-form) and
+0.002 (MAC) -- inside noise for all three. The gain from 4x depth is never positive beyond
noise for any model in either condition.

**Two cautions on reading this.** First, the *magnitude* is not robust: the same nominal
TRAIL configuration degrades by 0.012 in the main table and by 0.111 here, differing only in
training budget (12 epochs / 24k against 10 epochs / 20k). Only the **direction** is
consistent, and that is all we claim. Second, the `drop` column flatters the weakest model:
free-form latent recursion loses least (0.086) only because it starts at 0.53 and has less
to lose. On absolute out-of-distribution accuracy at native depth, TRAIL (0.658) is well
ahead of MAC (0.503) and free-form (0.438).

**What this means for the method.** Theorem B governs iterations *within* a hop and the
tests confirm the inner loop converges. Nothing constrains the *outer* sequence of hops, and
this is direct evidence that it does not converge -- running the same operator more times
moves the belief away from the answer rather than refining it. That is a sharper open
problem than the one we set out with, and it is the honest state of the work: TRAIL
generalises better than every baseline at its native depth, and it cannot buy more
generalisation with more compute.

### 5.7 (measured) The accuracy claim does not survive a matched comparison

The main table reported TRAIL ahead of every baseline by 19-22 points at a matched
*budget*. The ablations then showed that margin is carried entirely by the
control-conditioned relative-position bias inside the transport module (removing it costs
20.9 points and lands TRAIL in the baseline band) -- a component **no baseline had**. The
comparison was matched in belief updates and unmatched in inductive bias.

We therefore gave `attn1`, free-form latent recursion and MAC the same
`RelationalTransport` module, propagating their attention through it exactly as TRAIL
propagates belief. Each gains the identical 23,012 parameters, leaving mirror descent on the
simplex as the only remaining difference.

| model | no prior | + prior | change | belief updates |
|---|---|---|---|---|
| single cross-attention | 0.619 | **0.896** | +0.277 | **1** |
| MAC | 0.609 | **0.905** | +0.296 | 12 |
| free-form latent | 0.602 | 0.721 | +0.119 | 12 |
| TRAIL | 0.812 | 0.812 | -- | 4.32 |

**The accuracy claim is withdrawn.** It is not narrowed: it is reversed. MAC with a matched
prior beats TRAIL by 9.3 points, and a *single* relational attention step beats it by 8.4
while spending one belief update against TRAIL's twelve. By Prop. 1 attention is one TRAIL
step, so the T=1 case of the method outperforms the method.

**And the benchmark is the deeper problem.** If one application of a dense learned transport
reaches 89.6%, HOPWORLD does not require iterated reasoning at all, and therefore cannot
test the hypothesis this paper is about. The mechanism was visible earlier -- Sec. 5.3
diagnosed the halting failure as `M_h` composing several task hops in one application -- but
we read it as an explanation for that single failure rather than as evidence that the
testbed was inadequate. The benchmark is ours, so this is a design error and not a property
of an inherited task.

Three independent measurements now agree: iteration buys nothing (attn1 within 0.9 of MAC at
a twelfth of the compute), iterating longer hurts (Sec. 5.5), and constraining the iteration
to the simplex hurts more (this section).

### 5.8 What the paper can still claim

Everything below is measured, and none of it depends on the withdrawn comparison.

| claim | evidence |
|---|---|
| The inner certificate halves compute for free | 12 -> 6.32 belief updates at identical accuracy (0.8123, unchanged to 4 d.p.). Thm. A doing real work, no learned gate. |
| Zero off-manifold drift | 4x10^-4 against 10.76 for free-form recursion -- four orders of magnitude, as Prop. 2 requires. |
| The trace is the computation | p_t is the state, not a saliency map over it. |
| Anytime supervision earns its place in the cheap regime | +19 points over last-step training at 5 belief updates; the frontier is flat rather than cliffed. Cost: a ceiling ~6 points lower. |
| Attention is one step | Prop. 1, verified exactly in float64. |
| The two-loop structure | same accuracy as flat, at one third of the O(N^2) transport builds. |
| The theory | ten numerical checks, all passing. |

And the measured prices, which belong in the same table rather than a footnote: the
prescribed step size costs **7.7 points** against a learned one (which converges to eta =
0.961, violating Thm. B's eta <= 1/(4 beta) = 0.25); and the simplex constraint costs
**8.4 points** against a single unconstrained relational attention step.

The honest summary of this work is therefore: *latent visual reasoning can be made
on-manifold, certified, interpretable and adaptively cheap, and here is what that costs.*
That is a smaller claim than the one we set out to make, and it is the one the measurements
support.

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
