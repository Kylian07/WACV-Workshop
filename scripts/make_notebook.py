"""Regenerates notebooks/trail_wacv2027_kaggle.ipynb.

The notebook is generated rather than hand-edited so that prose and code cells
stay reviewable in a diff instead of buried in JSON.
"""
import json, pathlib

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def co(t): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":t.splitlines(keepends=True)}

cells = []

cells.append(md(r"""# TRAIL — Transport-Regularised Anytime Iterated Latent reasoning

**Target venue:** 1st Workshop on Latent Visual Reasoning (LVR), WACV 2027.

### The one-sentence idea

Current latent visual reasoning iterates a *free-floating* hidden state
`z ← z + f(z, c)` for a fixed number of steps. TRAIL replaces that recursion with
**entropic mirror descent on a question-conditioned energy defined over the simplex of
actual visual atoms**. One reasoning step becomes one optimisation step, and three
properties that are usually bolted on come out of the geometry for free:

| property | how current LVR gets it | how TRAIL gets it |
|---|---|---|
| staying on the visual manifold | hoped for / regularised | `z_t = Vᵀp_t ∈ conv(V)` by construction (Prop. 2) |
| knowing when to stop | fixed step count, or a learned gate | the Frank–Wolfe gap `G_t`, a *sound* bound on suboptimality (Thm. A) |
| interpretability | post-hoc saliency | `p_t` **is** the state — the trace is the computation |

and the step size is not a hyperparameter: Lemma 1 gives a **dimension-free** relative
smoothness modulus `L = 4β`, so `η = 1/(4β)` is prescribed.

### The claim this notebook is built to support

*Equal or better accuracy at a fraction of the reasoning steps, with a decodable trace
and markedly better generalisation to questions that need more hops than anything seen
in training.* Not "a new number on a saturated benchmark".

### What runs here

1. **Theory checks** — ten numerical tests, one per formal claim. ~1 minute.
2. **HOPWORLD** — a controlled *k*-hop task where ground-truth hop count is known exactly. Minutes.
3. **CLEVR** — the real study, with the depth-extrapolation and held-out-combination splits.

> **Runtime:** turn on the GPU accelerator (T4 ×2 or P100) and **Internet** (needed for the
> ImageNet trunk weights and, if you use it, the repo clone)."""))

cells.append(md("## 0 · Bootstrap\n\nFinds the `trail` package: local checkout → attached Kaggle dataset → git clone."))
cells.append(co(r'''import os, sys, subprocess, importlib, pathlib, json

REPO_URL = "https://github.com/Kylian07/WACV-Workshop.git"   # <- your fork
CANDIDATES = ["/kaggle/working/WACV-Workshop", ".", "/kaggle/working"]

def _find_pkg():
    for base in CANDIDATES:
        if pathlib.Path(base, "trail", "models", "trail_cell.py").exists():
            return base
    for base in pathlib.Path("/kaggle/input").glob("*"):          # attached as a dataset
        for hit in base.glob("**/trail/models/trail_cell.py"):
            return str(hit.parents[2])
    return None

root = _find_pkg()
if root is None:
    print("cloning ...")
    subprocess.run(["git", "clone", "-q", REPO_URL, "/kaggle/working/WACV-Workshop"], check=True)
    root = "/kaggle/working/WACV-Workshop"
sys.path.insert(0, root)
print("package root:", root)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")'''))

cells.append(md("""## 1 · The theory, checked numerically

Each test fails if its proposition is false. This is the artifact a reviewer can run in
under a minute — it is also how you catch a broken refactor before burning GPU hours."""))
cells.append(co('''import subprocess, sys
print(subprocess.run([sys.executable, f"{root}/tests/test_theory.py"],
                     capture_output=True, text=True).stdout)'''))

cells.append(md(r"""## 2 · HOPWORLD — the controlled study

A 7×7 grid of objects; a question names a start object by colour, then gives *k* direction
hops; the answer is an attribute of the object reached. **`k` is known exactly and is never
shown to the model**, which is what makes the halting analysis clean.

Three things are measured here that CLEVR cannot measure as cleanly:

* `ρ` — Spearman correlation between the step at which the certificate fired and the true `k`;
* **depth extrapolation** — train on `k ∈ {1,2,3}`, test on `k ∈ {4,5,6}`, and let TRAIL
  simply *iterate longer* at test time (something a fixed-depth recurrence cannot do);
* **drift** — distance of the visual state from `conv(V)`, identically zero for TRAIL."""))
cells.append(co('''QUICK = True          # False reproduces the paper's Table 1 (~30 min on a T4)

import subprocess, sys
cmd = [sys.executable, f"{root}/scripts/run_synthetic.py", "--device", DEVICE,
       "--out", "/kaggle/working/runs/synthetic",
       "--models", "trail,latent_freeform,mac,attn1,film"]
cmd += ["--quick", "--epochs", "8", "--n-train", "24000", "--n-val", "3000"] if QUICK else ["--full"]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-6000:]);  print(p.stderr[-2000:] if p.returncode else "")'''))

cells.append(md("### Figures from the controlled study"))
cells.append(co('''from trail.analysis.figures import make_all
import json, matplotlib.pyplot as plt, matplotlib.image as mpimg

res_path = "/kaggle/working/runs/synthetic/results.json"
paths = make_all(res_path, "/kaggle/working/figs_synthetic", grid=7)
for p_ in paths:
    plt.figure(figsize=(7, 5)); plt.imshow(mpimg.imread(p_)); plt.axis("off"); plt.show()'''))

cells.append(md(r"""### Depth extrapolation from the saved checkpoints

Nothing is retrained. Each run's `last.pt` is shown 4-6 hop questions after being trained
only on 1-3, first at its native depth and then at 4x it. TRAIL can be asked to iterate
longer because its step size is prescribed and its energy does not depend on how many steps
you take; the baselines are asked the same question so the comparison is fair."""))
cells.append(co('''p = subprocess.run([sys.executable, f"{root}/scripts/eval_extrapolation.py",
                    "--runs", "/kaggle/working/runs/synthetic", "--device", DEVICE],
                   capture_output=True, text=True)
print(p.stdout[-4000:]); print(p.stderr[-2000:] if p.returncode else "")'''))

cells.append(md("### Tables, as Markdown and LaTeX"))
cells.append(co('''p = subprocess.run([sys.executable, f"{root}/scripts/report.py",
                    "--runs", "/kaggle/working/runs/synthetic",
                    "--out", "/kaggle/working/paper_tables"],
                   capture_output=True, text=True)
print(p.stdout[-6000:]); print(p.stderr[-2000:] if p.returncode else "")'''))

cells.append(md(r"""## 3 · CLEVR

**Attach a CLEVR v1.0 dataset to this notebook** (right panel → *Add Input* → search
`CLEVR`; e.g. `timoboz/clevr-dataset`). The loader *discovers* the directory rather than
hard-coding a slug, so any mirror with the standard layout works:

```
<root>/questions/CLEVR_{train,val}_questions.json
<root>/images/{train,val}/CLEVR_{split}_%06d.png
```

Stage 1 runs a frozen ResNet-18 trunk once and caches 14×14×256 atoms as float16
(~100 KB/image). Stage 2 trains every reasoner off that one cache, so the comparison
differs only in the reasoning module."""))
cells.append(co('''from trail.data.clevr import find_clevr_root
try:
    print("CLEVR found at:", find_clevr_root())
except FileNotFoundError as e:
    print(e)'''))

cells.append(co('''SMOKE = True     # True: a few minutes end-to-end, to prove the pipeline runs.
                 # False: the real run (see budget note in the next markdown cell).

import subprocess, sys
cmd = [sys.executable, f"{root}/scripts/run_clevr.py", "--device", DEVICE,
       "--split", "iid", "--out", "/kaggle/working/runs/clevr",
       "--models", "trail,latent_freeform,mac,film,attn1"]
cmd += ["--smoke"] if SMOKE else ["--epochs", "8", "--train-images", "20000",
                                  "--train-questions", "200000", "--val-questions", "30000"]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-8000:]);  print(p.stderr[-3000:] if p.returncode else "")'''))

cells.append(md(r"""### Budget

On one T4, with 20 k images / 200 k questions / 8 epochs:

| stage | cost |
|---|---|
| featurising 20 k train + 5 k val images | ~25 min, once, reused by every model |
| training one reasoner | ~35–50 min |
| five models | ~4 h |

That fits Kaggle's 12 h session and 30 h weekly GPU quota with room for the two
generalisation splits. Scale `--train-questions` first if you are short on time — the
ranking between methods is stable well below the full 700 k."""))

cells.append(md(r"""## 4 · Depth extrapolation — the headline generalisation result

Train only on questions whose CLEVR program has ≤ 10 nodes; evaluate only on questions
with ≥ 15. A fixed-depth recurrence has no recourse. TRAIL is given one: **run more
mirror-descent steps**, on the same weights, because the energy and the prescribed step
size do not depend on how many steps you take."""))
cells.append(co('''cmd = [sys.executable, f"{root}/scripts/run_clevr.py", "--device", DEVICE,
       "--split", "depth", "--out", "/kaggle/working/runs/clevr_depth",
       "--models", "trail,latent_freeform,mac"]
cmd += ["--smoke"] if SMOKE else ["--epochs", "8", "--train-questions", "200000",
                                  "--val-questions", "30000"]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-6000:]);  print(p.stderr[-3000:] if p.returncode else "")'''))

cells.append(md("""### Held-out attribute combination (compositional split)

Every question mentioning *red* **and** *cube* is removed from training and forms the
test set. Tests whether reasoning composes, rather than whether the answer was memorised."""))
cells.append(co('''cmd = [sys.executable, f"{root}/scripts/run_clevr.py", "--device", DEVICE,
       "--split", "hoc", "--out", "/kaggle/working/runs/clevr_hoc",
       "--models", "trail,latent_freeform,mac"]
cmd += ["--smoke"] if SMOKE else ["--epochs", "8", "--train-questions", "200000",
                                  "--val-questions", "20000"]
p = subprocess.run(cmd, capture_output=True, text=True)
print(p.stdout[-6000:]);  print(p.stderr[-3000:] if p.returncode else "")'''))

cells.append(md("## 5 · Figures for the paper"))
cells.append(co('''from trail.analysis.figures import make_all
import matplotlib.pyplot as plt, matplotlib.image as mpimg, glob

for run, grid in [("/kaggle/working/runs/clevr/results_iid.json", 14),
                  ("/kaggle/working/runs/clevr_depth/results_depth.json", 14)]:
    try:
        out = run.replace("results", "figs").rsplit("/", 1)[0] + "/figs"
        for p_ in make_all(run, out, grid=grid):
            plt.figure(figsize=(7, 5)); plt.imshow(mpimg.imread(p_)); plt.axis("off"); plt.show()
    except FileNotFoundError:
        print("not run yet:", run)'''))

cells.append(md(r"""### Reasoning traces

`p_t` reshaped to the 14×14 atom grid, one column per mirror-descent step. No saliency
method is involved — this *is* the model's state, which is the point of constraining the
reasoning to the simplex in the first place."""))
cells.append(co('''import numpy as np, glob, matplotlib.pyplot as plt
for f in sorted(glob.glob("/kaggle/working/runs/clevr*/traces_*_trail.npz"))[:1]:
    z = np.load(f)
    keys = sorted(z.files)[:4]
    T = z[keys[0]].shape[0]
    fig, axes = plt.subplots(len(keys), T, figsize=(1.5 * T, 1.6 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        for t in range(T):
            ax = axes[i][t]; ax.imshow(z[k][t].reshape(14, 14), cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(f"hop {t+1}", fontsize=9)
    plt.suptitle("TRAIL belief p_t over the atom grid"); plt.tight_layout(); plt.show()'''))

cells.append(md(r"""## 6 · Ablations that reviewers will ask for

Each isolates one load-bearing claim. Run the ones you have budget for; `--models trail`
with the flags below, or edit `Config` directly.

| ablation | what it removes | claim it tests |
|---|---|---|
| `beta=0` (`use_transport=False`) | the relational transport term | is the *relational* energy doing work, or is this just iterated attention? |
| `learn_eta=True` | the prescribed step size | does Lemma 1's `η = 1/(4β)` actually beat a learned one? |
| `anytime_loss='last'` | prefix supervision | this is what breaks the compute–accuracy frontier |
| `inner_steps=1` | the two-loop structure | flat per-step re-conditioning vs. optimising a fixed energy |
| `delta=0` | the outer halting rule | how much of the saving comes from adaptive depth |
| `delta=1` | the *separation* of the two rules — the gap now decides hops | the conflation this design avoids |"""))
cells.append(co('''from trail.config import Config
from trail.models.model import VQAModel
from trail.data.synthetic import hopworld_splits
from trail.train import train, evaluate
import json, torch

tr, va, vocab, n_ans, fd = hopworld_splits(12000, 2500, 7, 8, seed=0)
ABL = {
  "TRAIL (full)":         dict(),
  "no transport (b=0)":   dict(use_transport=False, beta=0.0),
  "learned step size":    dict(learn_eta=True),
  "last-step loss":       dict(anytime_loss="last"),
  "flat (inner=1)":       dict(n_steps=12, inner_steps=1),
  "no halting (fixed T)": dict(delta=0.0),
  "gap decides hops":     dict(delta=1.0),   # the conflation the paper warns about
}
rows = {}
for name, over in ABL.items():
    cfg = Config(dataset="synthetic", reasoner="trail", grid=7, d_vis=128, d_ctrl=128,
                 d_hidden=256, n_steps=4, inner_steps=4, epochs=8, batch_size=128, lr=1e-3,
                 warmup=100, max_q_len=12, num_workers=0, use_count=False, dropout=0.0,
                 amp=(DEVICE == "cuda"), out_dir=f"/kaggle/working/runs/abl", **over)
    torch.manual_seed(0)
    m = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=fd)
    train(m, tr, va, cfg, device=DEVICE, log=lambda *a: None)
    e = evaluate(m, va, cfg, device=DEVICE)
    rows[name] = dict(acc=round(e["acc_full"], 4), acc_eps=round(e["acc_eps"], 4),
                      steps=round(e["avg_steps"], 2), updates=round(e["avg_updates"], 2),
                      rho=e.get("spearman_halt_hops"))
    print(f"{name:>22}  acc {rows[name]['acc']:.3f}  acc@cert {rows[name]['acc_eps']:.3f}  "
          f"hops {rows[name]['steps']:.2f}  updates {e['avg_updates']:.2f}")
json.dump(rows, open("/kaggle/working/runs/ablations.json", "w"), indent=2)'''))

cells.append(md(r"""## 7 · What goes in the paper

**Table 1** HOPWORLD: accuracy, average steps, ρ(halt, hops), OOD at *T* and at 2*T*.
**Table 2** CLEVR iid / depth / held-out-combination, all models at matched budget.
**Table 3** the ablations above.
**Fig. 2** certified gap `G_t` falling. **Fig. 3** drift. **Fig. 4** the compute–accuracy
frontier — TRAIL a curve, every baseline a point. **Fig. 6** halting step vs ground-truth
hops. **Fig. 7** traces.

Two honesty notes that belong in the paper and will save you a reviewer:

1. Numbers here are **not** comparable to published CLEVR results (MAC ≈ 98.9, NS-VQA ≈ 99.8)
   — those use ResNet-101 conv4 features at full resolution and days of training. Every model
   in this study shares one frozen ResNet-18 cache and one budget; the comparison is internal
   and should be described that way.
2. Thm. A gives ε-optimality **of the hop's energy**, not of the answer. The bridge from
   "this hop is solved" to "the answer is right" is empirical, and Sec. 5.3 is where it is
   tested. Say so before a reviewer does."""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
pathlib.Path("notebooks").mkdir(exist_ok=True)
json.dump(nb, open("notebooks/trail_wacv2027_kaggle.ipynb", "w"), indent=1)
print("wrote notebook with", len(cells), "cells")
