"""Section 5.1: the controlled HOPWORLD study.

Trains TRAIL and every baseline under an identical budget and reports the four
numbers the paper's claims reduce to:

  acc         accuracy in distribution
  steps       average reasoning steps actually spent at the certificate threshold
  rho         Spearman correlation between halting step and ground-truth hops
  ood         accuracy on questions needing more hops than anything seen in training

Usage:
    python scripts/run_synthetic.py --quick          # ~minutes on CPU
    python scripts/run_synthetic.py --full --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from trail.config import Config
from trail.data.synthetic import hopworld_splits
from trail.models.model import VQAModel
from trail.train import evaluate, measure_drift, train


def build(cfg, vocab, n_ans, feat_dim, device):
    return VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=feat_dim).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/synthetic")
    ap.add_argument("--models", default="trail,latent_freeform,mac,attn1,film")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=0)
    # 4 hops x 3 mirror-descent steps = 12 belief updates. The baselines are given
    # the same 12 (model.py multiplies them out), which is also MAC's standard depth.
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--inner", type=int, default=3)
    args = ap.parse_args()

    n_train, n_val, epochs = (4000, 1000, 4) if args.quick else (40000, 6000, 12)
    if args.epochs:
        epochs = args.epochs
    n_train = args.n_train or n_train
    n_val = args.n_val or n_val
    grid, n_obj = 7, 8

    tr, va, vocab, n_ans, feat_dim = hopworld_splits(
        n_train, n_val, grid, n_obj, train_hops=(1, 2, 3), test_hops=(1, 2, 3), seed=args.seed)
    _, ood, _, _, _ = hopworld_splits(
        100, n_val, grid, n_obj, train_hops=(1,), test_hops=(4, 5, 6), seed=args.seed + 7)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name in args.models.split(","):
        cfg = Config(
            dataset="synthetic", reasoner=name, features="cached", grid=grid,
            n_steps=args.hops, inner_steps=args.inner, beta=1.0, d_vis=128, d_ctrl=128,
            d_hidden=256, epochs=epochs, batch_size=128, lr=1e-3, warmup=100,
            max_q_len=12, eps=0.10, out_dir=str(Path(args.out) / name),
            num_workers=0, amp=args.device.startswith("cuda"), seed=args.seed,
            use_count=False, dropout=0.0,
        )
        torch.manual_seed(cfg.seed)
        model = build(cfg, vocab, n_ans, feat_dim, args.device)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t0 = time.time()
        print(f"\n=== {name}  ({n_par/1e6:.2f}M params) ===")
        train(model, tr, va, cfg, device=args.device)

        iid = evaluate(model, va, cfg, device=args.device)
        # depth extrapolation: the same model, allowed to keep iterating
        deep = evaluate(model, ood, cfg, device=args.device)
        deep_more = evaluate(model, ood, cfg, device=args.device, n_steps=16)
        drift = measure_drift(model, va, cfg, device=args.device)

        results[name] = {
            "params": n_par,
            "train_time_s": round(time.time() - t0, 1),
            "iid_acc": iid["acc_full"],
            "iid_acc_eps": iid["acc_eps"],
            "iid_steps": iid["avg_steps"],
            "iid_updates": iid["avg_updates"],
            "updates_per_step": iid["updates_per_step"],
            "spearman_halt_hops": iid.get("spearman_halt_hops"),
            "halt_vs_hops": iid.get("halt_vs_hops"),
            "halt_stats_by_delta": iid.get("halt_stats_by_delta"),
            "acc_by_hops": iid["acc_by_hops"],
            "ood_acc_sameT": deep["acc_full"],
            "ood_acc_moreT": deep_more["acc_full"],
            "frontier": iid["frontier"],
            "gap_curve": iid["gap_curve"],
            "inner_gap_curve": iid.get("inner_gap_curve"),
            "drift_per_step": drift,
        }
        print(json.dumps({k: v for k, v in results[name].items()
                          if k not in ("frontier", "gap_curve", "inner_gap_curve",
                                       "drift_per_step", "halt_vs_hops",
                                       "halt_stats_by_delta")}, indent=2))
        json.dump(results, open(Path(args.out) / "results.json", "w"), indent=2)

    print("\n" + "=" * 78)
    hdr = f"{'model':>16} {'params':>8} {'acc':>7} {'acc@eps':>8} {'steps':>6} {'rho':>6} {'OOD(T)':>7} {'OOD(2T)':>8}"
    print(hdr)
    print("-" * 78)
    for k, v in results.items():
        rho = v["spearman_halt_hops"]
        print(f"{k:>16} {v['params']/1e6:>7.2f}M {v['iid_acc']:>7.3f} {v['iid_acc_eps']:>8.3f} "
              f"{v['iid_steps']:>6.2f} {('n/a' if rho is None else f'{rho:.2f}'):>6} "
              f"{v['ood_acc_sameT']:>7.3f} {v['ood_acc_moreT']:>8.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
