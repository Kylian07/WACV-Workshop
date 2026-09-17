"""Sections 5.2-5.4: the CLEVR study, sized for one Kaggle session.

Stage 1 featurises the images we need with a frozen trunk and caches them.
Stage 2 trains each reasoner off that shared cache, so the only thing that
differs between rows of the table is the reasoning module.

    python scripts/run_clevr.py --split iid   --models trail,latent_freeform,mac,film
    python scripts/run_clevr.py --split depth --models trail,latent_freeform
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
from trail.data.clevr import (ClevrFeatures, build_index, build_vocab,
                              cache_features, find_clevr_root)
from trail.models.model import VQAModel
from trail.train import evaluate, measure_drift, train


def prepare(cfg, device):
    root = find_clevr_root(cfg.data_root)
    print(f"CLEVR root: {root}")
    tr_recs = build_index(root, cfg, "train")
    va_recs = build_index(root, cfg, "val")
    if cfg.max_train_images:
        keep = set(sorted({r["image_index"] for r in tr_recs})[: cfg.max_train_images])
        tr_recs = [r for r in tr_recs if r["image_index"] in keep]
    print(f"questions: train {len(tr_recs)}  val {len(va_recs)}")
    if not tr_recs or not va_recs:
        raise RuntimeError("empty split -- loosen Config.split / depth thresholds")

    # vocabulary from train only; answers from the union so val answers are scorable
    stoi, atoi = build_vocab(tr_recs)
    for r in va_recs:
        if r["answer"] not in atoi:
            atoi[r["answer"]] = len(atoi)

    tr_f, tr_pos, ch = cache_features(root, cfg, {r["image_index"] for r in tr_recs},
                                      "train", device=device)
    va_f, va_pos, _ = cache_features(root, cfg, {r["image_index"] for r in va_recs},
                                     "val", device=device)
    tr = ClevrFeatures(tr_recs, tr_f, tr_pos, stoi, atoi, cfg.max_q_len)
    va = ClevrFeatures(va_recs, va_f, va_pos, stoi, atoi, cfg.max_q_len)
    return tr, va, len(stoi), len(atoi), ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="trail,latent_freeform,mac,film,attn1")
    ap.add_argument("--split", default="iid", choices=["iid", "depth", "hoc"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="/kaggle/working/runs/clevr")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train-images", type=int, default=20000)
    ap.add_argument("--train-questions", type=int, default=200000)
    ap.add_argument("--val-questions", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    base = dict(
        dataset="clevr", data_root=args.data_root, split=args.split,
        features="resnet18", image_size=224, grid=14,
        max_train_images=args.train_images, max_train_questions=args.train_questions,
        max_val_questions=args.val_questions, epochs=args.epochs,
        batch_size=args.batch_size, n_steps=args.hops, inner_steps=args.inner,
    )
    if args.smoke:
        base.update(max_train_images=200, max_train_questions=3000, max_val_questions=1500,
                    epochs=1, batch_size=32, n_steps=3, inner_steps=2)

    prep_cfg = Config(**base, reasoner="trail", out_dir=args.out)
    tr, va, vocab, n_ans, ch = prepare(prep_cfg, args.device)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name in args.models.split(","):
        cfg = Config(**base, reasoner=name, out_dir=str(Path(args.out) / f"{args.split}_{name}"))
        os.makedirs(cfg.out_dir, exist_ok=True)
        cfg.save(Path(args.out) / f"cfg_{args.split}_{name}.json")
        torch.manual_seed(cfg.seed)
        model = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=ch)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== {name} on CLEVR/{args.split}  ({n_par/1e6:.2f}M params) ===")
        t0 = time.time()
        train(model, tr, va, cfg, device=args.device)
        m = evaluate(model, va, cfg, device=args.device, collect_traces=8)
        traces = m.pop("traces", [])
        results[name] = {
            "params": n_par, "train_time_s": round(time.time() - t0, 1),
            "iid_acc": m["acc_full"], "iid_acc_eps": m["acc_eps"], "iid_steps": m["avg_steps"],
            "spearman_halt_hops": m.get("spearman_halt_hops"),
            "acc_by_hops": m["acc_by_hops"], "frontier": m["frontier"],
            "gap_curve": m["gap_curve"], "halt_vs_hops": m.get("halt_vs_hops"),
            "drift_per_step": measure_drift(model, va, cfg, device=args.device),
        }
        json.dump(results, open(Path(args.out) / f"results_{args.split}.json", "w"), indent=2)
        if traces:
            import numpy as np
            np.savez(Path(args.out) / f"traces_{args.split}_{name}.npz",
                     **{f"p{i}": t["p"] for i, t in enumerate(traces)})
        print(json.dumps({k: v for k, v in results[name].items()
                          if k not in ("frontier", "gap_curve", "drift_per_step", "halt_vs_hops")},
                         indent=2))

    print("\n" + "=" * 72)
    print(f"{'model':>16} {'params':>8} {'acc':>7} {'acc@eps':>8} {'steps':>6} {'rho':>6}")
    print("-" * 72)
    for k, v in results.items():
        rho = v["spearman_halt_hops"]
        print(f"{k:>16} {v['params']/1e6:>7.2f}M {v['iid_acc']:>7.3f} {v['iid_acc_eps']:>8.3f} "
              f"{v['iid_steps']:>6.2f} {('n/a' if rho is None else f'{rho:.2f}'):>6}")
    print("=" * 72)


if __name__ == "__main__":
    main()
