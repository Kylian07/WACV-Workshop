# TRAIL — Transport-Regularised Anytime Iterated Latent reasoning

Reference implementation for a submission to the **1st Workshop on Latent Visual Reasoning
(LVR), WACV 2027**.

---

## The idea

Latent visual reasoning iterates a hidden state in visual embedding space instead of
decoding intermediate tokens. The recursions in use are free-form — `z ← z + f(z, c)`,
repeated a fixed number of times. Three things are missing from that, and all three are
usually patched with extra machinery:

* nothing keeps `z` in the region of embedding space the encoder actually produces;
* nothing says when reasoning is finished (depth is a hyperparameter, or a learned gate);
* nothing makes the intermediate states readable.

TRAIL changes the **parameterisation**, not the regularisation. The reasoning state is a
distribution `p` over the image's own visual atoms, and one reasoning step is one entropic
**mirror-descent step** on a question-conditioned energy

```
E_h(p) = -⟨s_h, p⟩ + (β/2)·‖p − M_h·p‖²
```

with `s_h` scoring atoms against the current sub-question and `M_h` a **column-stochastic
relational transport operator** — `M_h p` moves belief along the relation the sub-question
names. The quadratic term asks the belief to be a fixed point of that relation, which is
what turns a re-weighting of evidence into a hop.

Then:

| | |
|---|---|
| **on-manifold** | `z = Vᵀp ∈ conv(V)` at every step — drift is zero by construction, not by penalty |
| **knows when to stop** | the Frank–Wolfe gap `G = ⟨∇E,p⟩ − min_i ∇E_i` upper-bounds true suboptimality: a *sound* certificate, `O(N)`, no learned gate. It governs iterations *within* a hop; a separate belief-stationarity rule `KL(p_h‖p_{h−1}) ≤ δ` governs how many hops — conflating the two stops early on exactly the hard questions |
| **readable** | `p_t` is a distribution over image regions — the trace **is** the computation |
| **no step-size tuning** | `E` is `4β`-smooth relative to the entropy *independently of N*, so `η = 1/(4β)` is prescribed |
| **backward compatible** | softmax cross-attention is exactly one step (`T=1, β=0, η=1`) |

The claim the code is built to support: **equal or better accuracy at a fraction of the
reasoning steps, with a decodable trace and better generalisation to questions needing more
hops than anything seen in training** — not a new number on a saturated benchmark.

## The theory, and how to check it in one minute

```bash
python tests/test_theory.py
```

Ten numerical tests, one per formal claim — they fail if the proposition is false.

| claim | statement | test |
|---|---|---|
| Lemma 1 | `max_ij |H_ij| ≤ 4β` for every `N`; `H ⪰ 0` so `E` is convex | `test_lemma1_dimension_free_smoothness` |
| Thm. A | `G(p) ≥ E(p) − min_Δ E`: halting at `G ≤ ε` is ε-optimal | `test_theorem_a_gap_is_an_upper_bound` |
| Thm. B | monotone descent at `η = 1/(4β)`; `E(p_K) − E* ≤ 4β log N / K` | `test_theorem_b_*` |
| Prop. 1 | attention = one TRAIL step | `test_prop1_attention_is_one_trail_step` |
| Prop. 2 | `dist(z_t, conv(V)) = 0`; free-form recursion drifts | `test_prop2_no_drift` |
| Prop. 3 | halting time `∝ 1/margin` — hard inputs get more compute | `test_prop3_margin_controls_halting` |
| — | the module's inner loop really is that mirror descent | `test_module_inner_loop_is_the_theory` |

Proofs: `paper/TRAIL_paper.md` §4.

## Running it

**Kaggle:** open `notebooks/trail_wacv2027_kaggle.ipynb`, enable GPU + Internet, attach a
CLEVR v1.0 dataset, run top to bottom. It starts in smoke mode (minutes) — flip `SMOKE` and
`QUICK` to `False` for the real run.

**Locally:**

```bash
pip install -r requirements.txt
python tests/test_theory.py                                  # the math
python scripts/run_synthetic.py --quick                      # controlled study, minutes
python scripts/run_clevr.py --split iid --smoke              # CLEVR pipeline check
python scripts/run_clevr.py --split iid --epochs 8           # the real run
python scripts/run_clevr.py --split depth                    # depth extrapolation
python scripts/run_clevr.py --split hoc                      # held-out red∧cube
```

## Datasets

**CLEVR v1.0** is the primary benchmark: it ships a ground-truth functional program with
every question, so *reasoning depth* is a label we can correlate against without ever
training on it. That is what makes the central unsupervised-halting result measurable at
all. On Kaggle, attach any mirror with the standard layout (e.g. `timoboz/clevr-dataset`) —
`find_clevr_root()` discovers

```
<root>/questions/CLEVR_{train,val}_questions.json
<root>/images/{train,val}/CLEVR_{split}_%06d.png
```

rather than hard-coding a slug, so mirrors with different folder nesting work. Set
`Config.data_root` if discovery fails. CLEVR's test split has no answers; we train on
train and report on val, as is standard.

**GQA** (`trail/data/gqa.py`) is the real-image second benchmark, exposing the same
interface. `len(q['semantic'])` plays the role of CLEVR's program length, so the hop-count
analysis transfers verbatim. Attach a GQA mirror and point `Config.data_root` at the folder
holding `{train,val}_balanced_questions.json`.

**HOPWORLD** (`trail/data/synthetic.py`) is generated in-process, needs no download, and is
where the halting and extrapolation claims are tested without confounds: the number of hops
a question requires is known exactly and is never shown to the model.

> The Kaggle dataset slugs above could not be verified from the sandbox this code was
> written in (kaggle.com is unreachable there), which is exactly why the loaders discover
> paths instead of hard-coding them. Check the slug in the Kaggle UI when you attach it.

## Layout

```
trail/theory/certificates.py   energy, gradient, FW gap, drift, smoothness — one function per claim
trail/models/trail_cell.py     the two loops: hops outside, mirror descent inside
trail/models/baselines.py      attn1 / free-form latent / MAC / FiLM, same backbone and head
trail/data/                    clevr.py (streaming reader, splits, feature cache), gqa.py, synthetic.py
trail/train.py                 anytime loss, ε-sweep evaluation, drift measurement
trail/analysis/figures.py      every figure in the paper
scripts/run_{synthetic,clevr}.py
paper/TRAIL_paper.md           draft with full proofs
```

## Compute budget (one T4)

| stage | cost |
|---|---|
| theory tests | ~1 min, CPU |
| HOPWORLD, all 5 models | ~20 min |
| featurising 20k+5k CLEVR images (once, shared) | ~25 min |
| training one reasoner on CLEVR | ~35–50 min |

Five models plus two generalisation splits fits Kaggle's 12 h session and 30 h weekly quota.
Reduce `--train-questions` first if short on time; the ranking is stable well below 700 k.

## Honest notes

1. Results are **not** comparable to published CLEVR numbers (MAC ≈ 98.9, NS-VQA ≈ 99.8):
   those use ResNet-101 conv4 features at full resolution and days of training. Every model
   here shares one frozen ResNet-18 cache and one budget. The comparison is internal and the
   paper says so.
2. Theorem A certifies ε-optimality **of the hop's energy**, not of the answer. The bridge
   from "this hop is solved" to "the answer is right" is empirical, and §5.3 is where it is
   tested.
3. Only the *visual* state is drift-free; the control unit is an unconstrained recurrence.
