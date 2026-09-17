"""Turn run json into the paper's tables.

Writes LaTeX and Markdown side by side so the numbers in the draft and the
numbers in the submission cannot drift apart -- paste the LaTeX, keep the
Markdown for reading.

    python scripts/report.py --runs runs/synthetic --out paper/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LBL = {"trail": "TRAIL (ours)", "latent_freeform": "Free-form latent recursion",
       "mac": "MAC", "attn1": "Single cross-attention", "film": "FiLM"}
ORDER = ["film", "attn1", "mac", "latent_freeform", "trail"]


def _fmt(x, nd=1, pct=True):
    if x is None or x != x:
        return "--"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def _updates(r):
    return r.get("iid_updates", r.get("iid_steps", 0) * r.get("updates_per_step", 1))


def main_table(res: dict) -> tuple[str, str]:
    """Accuracy, compute actually spent, and the unsupervised halting correlation."""
    rows = []
    for k in ORDER + [k for k in res if k not in ORDER]:
        if k not in res:
            continue
        r = res[k]
        rows.append((
            LBL.get(k, k),
            f"{r['params'] / 1e6:.2f}",
            _fmt(r.get("iid_acc")),
            _fmt(r.get("iid_acc_eps")),
            _fmt(_updates(r), nd=1, pct=False),
            _fmt(r.get("spearman_halt_hops"), nd=2, pct=False),
            _fmt(r.get("ood_acc_sameT")),
            _fmt(r.get("ood_acc_moreT")),
        ))
    head = ["Model", "Params (M)", "Acc", "Acc @ cert.", "Updates", "rho",
            "OOD @1x", "OOD @4x"]
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        md += "| " + " | ".join(r) + " |\n"

    tex = ("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule\n"
           + " & ".join(head).replace("rho", "$\\rho$") + " \\\\\n\\midrule\n")
    for r in rows:
        tex += " & ".join(r) + " \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    return md, tex


def frontier_table(res: dict) -> str:
    """One trained TRAIL model, swept over the certificate threshold."""
    f = res.get("trail", {}).get("frontier")
    if not f:
        return ""
    per = res["trail"].get("updates_per_step", 1)
    md = "| delta (outer rule) | hops | belief updates | accuracy |\n|---|---|---|---|\n"
    for d in sorted(f, key=lambda d: -d.get("delta", d.get("eps", 0))):
        u = d.get("updates", d["steps"] * per)
        md += (f"| {d.get('delta', d.get('eps', 0)):.2f} | {d['steps']:.2f} | "
               f"{u:.2f} | {100 * d['acc']:.1f} |\n")
    return md


def gap_table(res: dict) -> str:
    g = res.get("trail", {}).get("gap_curve")
    if not g:
        return ""
    md = "| hop | mean certified gap G_t |\n|---|---|\n"
    for i, v in enumerate(g, 1):
        md += f"| {i} | {v:.4f} |\n"
    return md


def drift_table(res: dict) -> str:
    md = "| model | drift at final step |\n|---|---|\n"
    any_row = False
    for k in ORDER:
        d = res.get(k, {}).get("drift_per_step")
        if d:
            md += f"| {LBL.get(k, k)} | {d[-1]:.4f} |\n"
            any_row = True
    return md if any_row else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/synthetic")
    ap.add_argument("--results", default=None, help="explicit results json path")
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()

    path = Path(args.results) if args.results else Path(args.runs) / "results.json"
    res = json.load(open(path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md, tex = main_table(res)
    extra = {"Frontier (one model, swept eps)": frontier_table(res),
             "Certified gap per hop": gap_table(res),
             "Off-manifold drift": drift_table(res)}

    body = f"# Results: {path}\n\n## Main table\n\n{md}\n"
    for title, t in extra.items():
        if t:
            body += f"\n## {title}\n\n{t}\n"

    for name, extra_path in (("extrapolation", "extrapolation.json"),
                             ("ablations", "ablations.json")):
        p = Path(args.runs) / extra_path
        if p.exists():
            d = json.load(open(p))
            body += f"\n## {name.title()}\n\n| run | " + \
                    " | ".join(next(iter(d.values())).keys()) + " |\n|" + \
                    "---|" * (1 + len(next(iter(d.values())))) + "\n"
            for k, v in d.items():
                body += f"| {k} | " + " | ".join(
                    ("--" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x))
                    for x in v.values()) + " |\n"

    (out / "tables.md").write_text(body)
    (out / "main_table.tex").write_text(tex)
    print(body)
    print(f"\nwrote {out / 'tables.md'} and {out / 'main_table.tex'}")


if __name__ == "__main__":
    main()
