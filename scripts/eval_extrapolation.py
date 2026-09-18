"""Depth extrapolation, evaluated from saved checkpoints.

Trains nothing.  Loads each run's ``last.pt`` and asks the one question the
paper's generalisation claim reduces to:

    a model trained only on 1-3 hop questions is shown 4-6 hop questions.
    Does letting it *iterate longer* -- same weights, nothing retrained --
    recover accuracy?

TRAIL can be asked that question because its step size is prescribed rather than
tuned and its energy does not depend on how many steps you take; a fixed-depth
recurrence can be asked it too, which is why every model here is run at 1x and
4x its native depth.  The budget is expressed in *belief updates* so the
comparison stays matched: TRAIL's H hops x K inner steps against a baseline's
H*K iterations.

    python scripts/eval_extrapolation.py --runs runs/synthetic
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
from trail.train import evaluate


def native_depth(model, cfg) -> int:
    """Belief updates the model performs at its trained depth."""
    if cfg.reasoner == "trail":
        return cfg.n_steps * cfg.inner_steps
    return int(getattr(model.reasoner, "n_steps", cfg.n_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/synthetic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--factor", type=int, default=4, help="test-time depth multiplier")
    args = ap.parse_args()

    tr, va, vocab, n_ans, feat_dim = hopworld_splits(100, args.n_val, 7, 8,
                                                     train_hops=(1, 2, 3), test_hops=(1, 2, 3))
    _, ood, _, _, _ = hopworld_splits(100, args.n_val, 7, 8,
                                      train_hops=(1,), test_hops=(4, 5, 6), seed=7)

    rows = {}
    for ck in sorted(Path(args.runs).glob("*/last.pt")):
        name = ck.parent.name
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = Config(**{k: v for k, v in blob["cfg"].items() if k in Config.__dataclass_fields__})
        model = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=feat_dim)
        model.load_state_dict(blob["model"])
        model.to(args.device)

        base = cfg.n_steps if cfg.reasoner == "trail" else native_depth(model, cfg)
        deep = base * args.factor
        iid = evaluate(model, va, cfg, device=args.device)
        o1 = evaluate(model, ood, cfg, device=args.device)
        o4 = evaluate(model, ood, cfg, device=args.device, n_steps=deep)
        rows[name] = {
            "iid_acc": iid["acc_full"],
            "ood_acc_native": o1["acc_full"],
            f"ood_acc_{args.factor}x": o4["acc_full"],
            "delta": o4["acc_full"] - o1["acc_full"],
            "updates_native": native_depth(model, cfg),
            "updates_deep": native_depth(model, cfg) * args.factor,
        }
        print(f"{name:>16}  iid {rows[name]['iid_acc']:.3f}  "
              f"ood@native {rows[name]['ood_acc_native']:.3f}  "
              f"ood@{args.factor}x {rows[name][f'ood_acc_{args.factor}x']:.3f}  "
              f"(delta {rows[name]['delta']:+.3f})")

    json.dump(rows, open(Path(args.runs) / "extrapolation.json", "w"), indent=2)
    print("\nwrote", Path(args.runs) / "extrapolation.json")


if __name__ == "__main__":
    main()
