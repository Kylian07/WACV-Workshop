"""Every figure in the paper, generated from the json that train/eval write.

Kept deliberately plain: one figure per function, no styling that depends on a
theme, so the same calls work in a Kaggle notebook and in the LaTeX build.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# colour-blind-safe, consistent across every figure
C = {"trail": "#0B6E4F", "latent_freeform": "#C1666B", "mac": "#4C6EF5",
     "attn1": "#8D8D92", "film": "#E8A33D"}
LBL = {"trail": "TRAIL (ours)", "latent_freeform": "free-form latent",
       "mac": "MAC", "attn1": "single attention", "film": "FiLM"}


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def fig_frontier(results: dict, out: Path):
    """Fig. 4: accuracy vs reasoning steps.

    TRAIL contributes a *curve* traced by sweeping the single threshold eps on
    one trained model; every fixed-depth baseline contributes a point.  That
    asymmetry is the compute-adaptivity claim in one picture.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    def _updates(r, key="iid_steps"):
        return r.get("iid_updates", r[key] * r.get("updates_per_step", 1))

    for name, r in results.items():
        if name == "trail" and r.get("frontier"):
            f = sorted(r["frontier"], key=lambda d: d.get("updates", d["steps"]))
            ax.plot([d.get("updates", d["steps"]) for d in f], [100 * d["acc"] for d in f],
                    "-o", color=C[name], ms=4, lw=1.8, label=LBL[name])
        else:
            ax.scatter([_updates(r)], [100 * r["iid_acc"]], color=C.get(name, "k"),
                       marker="s", s=45, label=LBL.get(name, name), zorder=3)
    _style(ax, "average belief updates (matched unit across models)", "accuracy (%)",
           "One model, a whole compute-accuracy frontier")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_halt_vs_hops(halt_vs_hops, out: Path, rho=None):
    """Fig. 6: the certificate discovers question difficulty without supervision."""
    a = np.asarray(halt_vs_hops)
    halt, hops = a[:, 0] + 1, a[:, 1]
    ks = sorted(set(hops.tolist()))
    data = [halt[hops == k] for k in ks]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    parts = ax.violinplot(data, positions=range(len(ks)), showmeans=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(C["trail"])
        pc.set_alpha(0.45)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(int(k)) for k in ks])
    t = "Halting step vs ground-truth hops"
    if rho is not None and rho == rho:
        t += f"   (Spearman rho = {rho:.2f})"
    _style(ax, "ground-truth reasoning hops (never shown to the model)",
           "steps taken before the certificate fired", t)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_drift(results: dict, out: Path):
    """Fig. 3: distance of the visual state from conv(V) along the trajectory."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name, r in results.items():
        d = r.get("drift_per_step")
        if not d:
            continue
        ax.plot(range(1, len(d) + 1), d, "-o", ms=3, color=C.get(name, "k"),
                label=LBL.get(name, name))
    _style(ax, "reasoning step", r"dist$(z_t,\ \mathrm{conv}(V))$",
           "Off-manifold drift (zero for TRAIL by Prop. 2)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_gap_curve(results: dict, out: Path):
    """Fig. 2: the certified suboptimality bound falling, averaged over the test set."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    g = results.get("trail", {}).get("gap_curve")
    if g:
        ax.semilogy(range(1, len(g) + 1), np.maximum(g, 1e-8), "-o", ms=4, color=C["trail"])
    _style(ax, "reasoning step $t$", r"mean Frank-Wolfe gap $G_t$",
           r"Certified bound on $E(p_t)-E^\star$")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_traces(traces, grid: int, out: Path, n: int = 4):
    """Fig. 7: the reasoning trace, decoded.

    Each row is one question; each column one mirror-descent step.  There is no
    saliency method here -- p_t *is* the model's state, so the picture is the
    computation rather than an explanation of it.
    """
    n = min(n, len(traces))
    T = min(6, traces[0]["p"].shape[0])
    fig, axes = plt.subplots(n, T, figsize=(1.35 * T, 1.45 * n), squeeze=False)
    for i in range(n):
        for t in range(T):
            ax = axes[i][t]
            ax.imshow(traces[i]["p"][t].reshape(grid, grid), cmap="magma")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"$p_{{{t}}}$", fontsize=9)
            if t == 0:
                ax.set_ylabel(f"{traces[i]['hops']} hops", fontsize=8)
    fig.suptitle("Decoded reasoning trace: the belief at every step", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def make_all(results_path: str, out_dir: str, grid: int = 7, traces=None):
    res = json.load(open(results_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_frontier(res, out / "fig4_frontier.png")
    fig_drift(res, out / "fig3_drift.png")
    fig_gap_curve(res, out / "fig2_gap.png")
    hv = res.get("trail", {}).get("halt_vs_hops")
    if hv:
        fig_halt_vs_hops(hv, out / "fig6_halt_vs_hops.png",
                         res["trail"].get("spearman_halt_hops"))
    if traces:
        fig_traces(traces, grid, out / "fig7_traces.png")
    return sorted(str(p) for p in out.glob("*.png"))
