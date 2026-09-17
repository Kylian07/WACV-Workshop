"""Figures and rendered equations for the TRAIL technical report.

Equations go through matplotlib's mathtext rather than being typed as Unicode:
ReportLab's built-in fonts have no glyphs for most maths, and DejaVu covers the
symbols but not the layout (fractions, sub-expressions under norms).  Rendering
to PNG keeps the report readable and the LaTeX source in one place.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

INK = "#1B2430"
GREEN = "#0B6E4F"
RED = "#C1666B"
BLUE = "#4C6EF5"
GREY = "#8D8D92"
AMBER = "#E8A33D"
PAPER = "#FFFFFF"


# --------------------------------------------------------------------------
def equation(latex: str, name: str, fontsize: int = 15, color: str = INK) -> Path:
    """Render one display equation to a tightly-cropped transparent PNG."""
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=color)
    fig.canvas.draw()
    bbox = t.get_window_extent()
    fig.set_size_inches(bbox.width / fig.dpi * 1.02, bbox.height / fig.dpi * 1.15)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, tc="white", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.35)


def _arrow(ax, p, q, color=INK, style="-|>", lw=1.3, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture() -> Path:
    """End-to-end data flow: where TRAIL sits and what it replaces."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    _box(ax, 0.05, 1.85, 1.5, 0.85, "image", GREY)
    _box(ax, 0.05, 0.45, 1.5, 0.85, "question", GREY)
    _box(ax, 1.85, 1.85, 1.9, 0.85, "frozen trunk\n$\\rightarrow$ atoms $V$", BLUE)
    _box(ax, 1.85, 0.45, 1.9, 0.85, "BiLSTM\n$\\rightarrow$ words, $q$", BLUE)
    _box(ax, 4.05, 0.9, 3.0, 1.75,
         "TRAIL reasoner\nouter: hops $h$ fix $E_h$\ninner: mirror descent on $E_h$",
         GREEN)
    _box(ax, 7.35, 1.55, 1.35, 0.85, "answer\nhead", BLUE)
    _box(ax, 7.35, 0.45, 1.35, 0.8, "certificates\n$G_k$, $\\mathrm{KL}$", AMBER, tc=INK)

    for y in (2.27, 0.87):
        _arrow(ax, (1.58, y), (1.82, y))
    _arrow(ax, (3.78, 2.27), (4.02, 2.0))
    _arrow(ax, (3.78, 0.87), (4.02, 1.3))
    _arrow(ax, (7.1, 1.95), (7.32, 1.95))
    _arrow(ax, (7.1, 1.2), (7.32, 0.9), color=AMBER)
    ax.text(5.55, 0.62, "state stays on $\\Delta^{N-1}$, so $z=V^{\\!\\top}p\\in\\mathrm{conv}(V)$",
            ha="center", fontsize=7.5, color=GREEN, style="italic")
    fig.tight_layout(pad=0.2)
    out = ASSETS / "fig_architecture.png"
    fig.savefig(out, dpi=300, facecolor=PAPER)
    plt.close(fig)
    return out


def fig_two_loops() -> Path:
    """The structure the theorems are about: energy fixed inside the inner loop."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.15, 0.2), 9.7, 3.6, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="#F4F6F8", ec=GREY, lw=1.0, zorder=0))
    ax.text(0.38, 3.52, "OUTER LOOP  $h=1\\ldots H$  (one hop = one sub-question)",
            fontsize=8.5, color=INK, weight="bold")

    _box(ax, 0.4, 2.45, 2.1, 0.8, "control unit\n$c_h$ from question words", BLUE, fs=8)
    _box(ax, 2.75, 2.45, 2.3, 0.8, "fix the energy $E_h$\n$s_h$, $M_h$ (col.-stochastic)", GREEN, fs=8)

    ax.add_patch(FancyBboxPatch((0.4, 0.5), 6.6, 1.65, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="white", ec=GREEN, lw=1.3, ls="--", zorder=1))
    ax.text(0.62, 1.92, "INNER LOOP  $k=1\\ldots K$  —  $E_h$ is HELD FIXED here",
            fontsize=8, color=GREEN, weight="bold", zorder=3)
    _box(ax, 0.65, 0.72, 1.85, 0.85, "$g_k=\\nabla E_h(p_k)$", GREEN, fs=8)
    _box(ax, 2.75, 0.72, 2.0, 0.85, "$p_{k+1}\\propto p_k e^{-\\eta g_k}$\n$\\eta=1/(4\\beta)$", GREEN, fs=8)
    _box(ax, 5.0, 0.72, 1.8, 0.85, "gap $G_k$\nstop if $\\leq\\varepsilon G_0$", AMBER, fs=8, tc=INK)

    _box(ax, 7.3, 1.2, 2.3, 1.3,
         "outer stop:\n$\\mathrm{KL}(p_h\\|p_{h-1})\\leq\\delta$\n(belief stopped moving)",
         AMBER, fs=8, tc=INK)

    _arrow(ax, (2.55, 2.85), (2.72, 2.85))
    _arrow(ax, (3.9, 2.42), (3.9, 2.2), color=GREEN)
    _arrow(ax, (2.52, 1.14), (2.72, 1.14), color=GREEN)
    _arrow(ax, (4.78, 1.14), (4.97, 1.14), color=GREEN)
    _arrow(ax, (5.9, 1.6), (5.9, 2.4), color=GREEN, rad=-0.45, ls=":")
    _arrow(ax, (7.02, 1.5), (7.27, 1.6), color=AMBER)
    ax.text(6.05, 2.0, "next $k$", fontsize=7, color=GREEN, style="italic")
    ax.text(0.4, 0.28, "Theorems apply because $E_h$ does not move during the inner loop.",
            fontsize=7.5, color=INK, style="italic")
    fig.tight_layout(pad=0.2)
    out = ASSETS / "fig_two_loops.png"
    fig.savefig(out, dpi=300, facecolor=PAPER)
    plt.close(fig)
    return out


def fig_frontier(frontier, baselines=None) -> Path:
    """Measured compute-accuracy frontier from a single trained model."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    f = sorted(frontier, key=lambda d: d["updates"])
    ax.plot([d["updates"] for d in f], [100 * d["acc"] for d in f], "-o",
            color=GREEN, ms=5, lw=2, label="TRAIL (one model, swept $\\delta$)")
    for d in f:
        if d["delta"] in (0.0, 0.01, 0.05):
            ax.annotate(f"$\\delta$={d['delta']:g}", (d["updates"], 100 * d["acc"]),
                        textcoords="offset points", xytext=(6, -9), fontsize=7.5, color=GREEN)
    for name, r in (baselines or {}).items():
        ax.scatter([r["updates"]], [100 * r["acc"]], marker="s", s=48,
                   color={"latent_freeform": RED, "mac": BLUE, "film": AMBER,
                          "attn1": GREY}.get(name, GREY), zorder=3, label=r["label"])
    ax.set_xlabel("average belief updates spent")
    ax.set_ylabel("accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(pad=0.4)
    out = ASSETS / "fig_frontier.png"
    fig.savefig(out, dpi=300, facecolor=PAPER)
    plt.close(fig)
    return out


def fig_halting_saturation(rows) -> Path:
    """The negative result: the outer rule saturates, so rho has no variance to use."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    d = [r[0] for r in rows]
    x = np.arange(len(d))
    a1.bar(x, [r[2] for r in rows], color=GREEN, alpha=0.85)
    a1.set_xticks(x)
    a1.set_xticklabels([f"{v:g}" for v in d], fontsize=7.5)
    a1.set_xlabel("$\\delta$ (outer threshold)")
    a1.set_ylabel("std. dev. of halting step")
    a1.set_title("Spread of the halting step", fontsize=9, loc="left")

    rho = [np.nan if r[3] is None else r[3] for r in rows]
    a2.bar(x, rho, color=RED, alpha=0.85)
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xticks(x)
    a2.set_xticklabels([f"{v:g}" for v in d], fontsize=7.5)
    a2.set_ylim(-0.1, 1.0)
    a2.set_xlabel("$\\delta$ (outer threshold)")
    a2.set_ylabel(r"$\rho$(halt, true hops)")
    a2.set_title("Correlation with ground-truth depth", fontsize=9, loc="left")
    for i, r in enumerate(rows):
        if r[3] is None:
            a2.text(i, 0.04, "undef.", ha="center", fontsize=6.5, color=GREY, rotation=90)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, lw=0.5, axis="y")
    fig.tight_layout(pad=0.4)
    out = ASSETS / "fig_halting.png"
    fig.savefig(out, dpi=300, facecolor=PAPER)
    plt.close(fig)
    return out


def fig_repo_map() -> Path:
    """What lives where, grouped by what it is responsible for."""
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    groups = [
        (0.1, GREEN, "the method", [
            "trail/theory/certificates.py   energy, gradient, FW gap, drift, smoothness",
            "trail/models/trail_cell.py     the two loops (hops / mirror descent)",
            "trail/models/heads.py          read-out incl. soft cardinality",
        ]),
        (0.1, BLUE, "the comparison", [
            "trail/models/baselines.py      attn1, free-form latent, MAC, FiLM",
            "trail/models/{backbone,text,model}.py   shared front end",
            "trail/train.py                 anytime loss, delta-sweep eval, drift",
        ]),
        (0.1, AMBER, "the data", [
            "trail/data/clevr.py            streaming reader, splits, depth labels",
            "trail/data/gqa.py              real images, same interface",
            "trail/data/synthetic.py        HOPWORLD (exact hop counts)",
            "trail/data/features.py         shared frozen-trunk cache",
        ]),
        (0.1, GREY, "evidence and delivery", [
            "tests/test_theory.py           10 checks, one per formal claim",
            "tests/test_eval_budget.py      matched-budget regression guard",
            "tests/test_clevr_io.py         loader against a hostile fixture",
            "scripts/run_*.py, report.py    experiments and paper tables",
            "notebooks/...kaggle.ipynb      the runnable submission artefact",
        ]),
    ]
    y = 4.72
    for _, color, title, lines in groups:
        ax.add_patch(FancyBboxPatch((0.1, y - 0.30 * len(lines) - 0.30), 9.8,
                                    0.30 * len(lines) + 0.34,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc="white", ec=color, lw=1.2))
        ax.text(0.28, y - 0.03, title.upper(), fontsize=8, color=color, weight="bold")
        for i, ln in enumerate(lines):
            ax.text(0.34, y - 0.33 - 0.30 * i, ln, fontsize=7.4, color=INK,
                    family="DejaVu Sans Mono")
        y -= 0.30 * len(lines) + 0.50
    fig.tight_layout(pad=0.2)
    out = ASSETS / "fig_repo_map.png"
    fig.savefig(out, dpi=300, facecolor=PAPER)
    plt.close(fig)
    return out
