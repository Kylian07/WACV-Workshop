"""Section 5.6: which part of TRAIL is doing the work?

The main table shows TRAIL well ahead of every baseline at a matched budget, but
it also shows something awkward: a *single* cross-attention beats both 12-step
recurrent baselines. So iteration on its own is not what helps, and the obvious
challenge is whether TRAIL's margin comes from iterating at all or just from
having a relational transport operator bolted to an attention block.

These ablations answer that directly. Each removes one load-bearing piece and
leaves everything else -- parameters, budget, schedule, data -- alone.

    python scripts/run_ablations.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from trail.config import Config
from trail.data.synthetic import hopworld_splits
from trail.models.model import VQAModel
from trail.train import evaluate, train

ABLATIONS = {
    "TRAIL (full)": ({}, "reference"),
    "no transport (beta=0)": ({"use_transport": False, "beta": 0.0},
                              "is the relational energy doing work, or is this iterated attention?"),
    "no position bias": ({"use_pos_bias": False},
                         "can transport find directions without the relative-position prior?"),
    "learned step size": ({"learn_eta": True},
                          "does Lemma 1's prescribed eta beat a learned one?"),
    "last-step loss": ({"anytime_loss": "last"},
                       "prefix supervision is what should make the frontier flat"),
    "flat (inner=1)": ({"n_steps": 12, "inner_steps": 1},
                       "two loops vs. re-conditioning the energy every step"),
    "fixed depth (delta=0)": ({"delta": 0.0},
                              "how much does adaptive depth cost or save?"),
    "gap decides hops (delta=1)": ({"delta": 1.0},
                                   "the conflation the two-rule design avoids"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/ablations")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-val", type=int, default=3000)
    args = ap.parse_args()

    tr, va, vocab, n_ans, fd = hopworld_splits(args.n_train, args.n_val, 7, 8, seed=0)
    os.makedirs(args.out, exist_ok=True)
    rows = {}
    for name, (over, why) in ABLATIONS.items():
        # Build the kwargs first and let the ablation override them. Splatting `over`
        # into a call that already names beta/n_steps/inner_steps is a TypeError the
        # moment an ablation touches one of those, which is exactly what they do.
        kw = dict(dataset="synthetic", reasoner="trail", grid=7, d_vis=128, d_ctrl=128,
                  d_hidden=256, n_steps=4, inner_steps=3, beta=1.0, epochs=args.epochs,
                  batch_size=128, lr=1e-3, warmup=100, max_q_len=12, num_workers=0,
                  use_count=False, dropout=0.0, amp=args.device.startswith("cuda"),
                  out_dir=str(Path(args.out) / name.replace(" ", "_")))
        kw.update(over)
        cfg = Config(**kw)
        torch.manual_seed(cfg.seed)
        m = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=fd)
        print(f"\n=== {name} === ({why})")
        train(m, tr, va, cfg, device=args.device, log=lambda *a: None)
        e = evaluate(m, va, cfg, device=args.device)
        rows[name] = {
            "why": why,
            "params": sum(p.numel() for p in m.parameters() if p.requires_grad),
            "acc_full": round(e["acc_full"], 4),
            "acc_cert": round(e["acc_eps"], 4),
            "hops": round(e["avg_steps"], 2),
            "updates": round(e["avg_updates"], 2),
        }
        print(f"  acc {rows[name]['acc_full']:.3f} | acc@cert {rows[name]['acc_cert']:.3f} "
              f"| hops {rows[name]['hops']:.2f} | updates {rows[name]['updates']:.2f}")
        json.dump(rows, open(Path(args.out) / "ablations.json", "w"), indent=2)

    ref = rows["TRAIL (full)"]["acc_full"]
    print("\n" + "=" * 84)
    print(f"{'ablation':>30} {'acc':>7} {'delta':>8} {'acc@cert':>9} {'updates':>8}")
    print("-" * 84)
    for k, v in rows.items():
        print(f"{k:>30} {v['acc_full']:>7.3f} {v['acc_full']-ref:>+8.3f} "
              f"{v['acc_cert']:>9.3f} {v['updates']:>8.2f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
