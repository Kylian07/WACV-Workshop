"""Section 5.5: does "just iterate longer" actually work?

The headline generalisation claim is that TRAIL can be handed questions needing
more hops than anything in training and recover accuracy by running more
mirror-descent steps on the same weights.  That claim is only *testable* if the
hops share one operator.  A model with a per-hop control embedding has four
learned sub-question slots and no way to express a fifth, so iterating it longer
cannot help and the experiment would measure the embedding table rather than the
method.  This script therefore trains with ``share_steps=True`` and reports both
conditions so the difference is visible rather than assumed.

Train on 1-3 hop HOPWORLD questions, test on 4-6, at native depth and at 4x.

    python scripts/run_extrapolation_study.py --device cuda
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="trail,latent_freeform,mac")
    ap.add_argument("--out", default="runs/extrapolation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--factor", type=int, default=4)
    args = ap.parse_args()

    tr, va, vocab, n_ans, fd = hopworld_splits(
        args.n_train, args.n_val, 7, 8, train_hops=(1, 2, 3), test_hops=(1, 2, 3), seed=0)
    _, ood, _, _, _ = hopworld_splits(
        100, args.n_val, 7, 8, train_hops=(1,), test_hops=(4, 5, 6), seed=7)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for share in (True, False):
        for name in args.models.split(","):
            key = f"{name}{'_tied' if share else '_untied'}"
            cfg = Config(
                dataset="synthetic", reasoner=name, grid=7, d_vis=128, d_ctrl=128,
                d_hidden=256, n_steps=args.hops, inner_steps=args.inner, beta=1.0,
                epochs=args.epochs, batch_size=128, lr=1e-3, warmup=100, max_q_len=12,
                num_workers=0, use_count=False, dropout=0.0, share_steps=share,
                amp=args.device.startswith("cuda"), out_dir=str(Path(args.out) / key),
            )
            torch.manual_seed(cfg.seed)
            model = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=fd)
            print(f"\n=== {key} ===")
            train(model, tr, va, cfg, device=args.device, log=lambda *a: None)

            native = cfg.n_steps if name == "trail" else cfg.n_steps * cfg.inner_steps
            iid = evaluate(model, va, cfg, device=args.device)
            o1 = evaluate(model, ood, cfg, device=args.device)
            o4 = evaluate(model, ood, cfg, device=args.device, n_steps=native * args.factor)
            results[key] = {
                "iid_acc": iid["acc_full"],
                "ood_native": o1["acc_full"],
                f"ood_{args.factor}x": o4["acc_full"],
                "gain_from_iterating": o4["acc_full"] - o1["acc_full"],
                "ood_drop": iid["acc_full"] - o4["acc_full"],
            }
            r = results[key]
            print(f"  iid {r['iid_acc']:.3f} | ood@native {r['ood_native']:.3f} "
                  f"| ood@{args.factor}x {r[f'ood_{args.factor}x']:.3f} "
                  f"| gain {r['gain_from_iterating']:+.3f}")
            json.dump(results, open(Path(args.out) / "extrapolation_study.json", "w"), indent=2)

    print("\n" + "=" * 76)
    print(f"{'run':>22} {'iid':>7} {'ood@1x':>8} {f'ood@{args.factor}x':>8} {'gain':>8} {'drop':>8}")
    print("-" * 76)
    for k, v in results.items():
        print(f"{k:>22} {v['iid_acc']:>7.3f} {v['ood_native']:>8.3f} "
              f"{v[f'ood_{args.factor}x']:>8.3f} {v['gain_from_iterating']:>+8.3f} "
              f"{v['ood_drop']:>8.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
