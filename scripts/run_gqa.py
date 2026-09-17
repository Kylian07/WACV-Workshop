"""GQA: the real-image counterpart of the CLEVR study.

Same protocol, same code path -- only the loader differs.  GQA's ``semantic``
field gives an operation chain per question, so the hop-count analysis of
Sec. 5.3 transfers verbatim: the number of ``relate`` operations plays the role
of CLEVR's relational program nodes.

    python scripts/run_gqa.py --models trail,latent_freeform,mac --epochs 6
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
from trail.data.gqa import (GqaFeatures, build_gqa_vocab, cache_gqa_features,
                            find_gqa_root, load_gqa_questions)
from trail.models.model import VQAModel
from trail.train import evaluate, measure_drift, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="trail,latent_freeform,mac")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="/kaggle/working/runs/gqa")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-questions", type=int, default=150000)
    ap.add_argument("--val-questions", type=int, default=20000)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    base = dict(dataset="gqa", data_root=args.data_root, features="resnet18",
                image_size=224, grid=14, max_q_len=32, epochs=args.epochs,
                n_steps=args.hops, inner_steps=args.inner)
    if args.smoke:
        base.update(epochs=1, n_steps=3, inner_steps=2, batch_size=32)
        args.train_questions, args.val_questions = 3000, 1500

    cfg0 = Config(**base, reasoner="trail", out_dir=args.out)
    root = find_gqa_root(cfg0.data_root)
    print("GQA questions root:", root)
    tr_recs = load_gqa_questions(root, "train", args.train_questions)
    va_recs = load_gqa_questions(root, "val", args.val_questions)
    print(f"questions: train {len(tr_recs)}  val {len(va_recs)}")

    stoi, atoi = build_gqa_vocab(tr_recs)
    # answers unseen in train are unanswerable by construction; they stay in the
    # denominator rather than being dropped, which is the honest accounting.
    tr_f, tr_pos, ch = cache_gqa_features(root, cfg0, {r["image_index"] for r in tr_recs},
                                          "train", device=args.device)
    va_f, va_pos, _ = cache_gqa_features(root, cfg0, {r["image_index"] for r in va_recs},
                                         "val", device=args.device)
    tr = GqaFeatures(tr_recs, tr_f, tr_pos, stoi, atoi, cfg0.max_q_len)
    va = GqaFeatures(va_recs, va_f, va_pos, stoi, atoi, cfg0.max_q_len)

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name in args.models.split(","):
        cfg = Config(**base, reasoner=name, out_dir=str(Path(args.out) / name))
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)
        model = VQAModel(cfg, vocab=len(stoi), n_answers=len(atoi), feat_dim=ch)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== {name} on GQA  ({n_par/1e6:.2f}M params) ===")
        t0 = time.time()
        train(model, tr, va, cfg, device=args.device)
        m = evaluate(model, va, cfg, device=args.device)
        results[name] = {
            "params": n_par, "train_time_s": round(time.time() - t0, 1),
            "iid_acc": m["acc_full"], "iid_acc_eps": m["acc_eps"],
            "iid_steps": m["avg_steps"], "iid_updates": m["avg_updates"],
            "updates_per_step": m["updates_per_step"],
            "spearman_halt_hops": m.get("spearman_halt_hops"),
            "acc_by_hops": m["acc_by_hops"], "frontier": m["frontier"],
            "gap_curve": m["gap_curve"], "inner_gap_curve": m.get("inner_gap_curve"),
            "halt_vs_hops": m.get("halt_vs_hops"),
            "drift_per_step": measure_drift(model, va, cfg, device=args.device),
        }
        json.dump(results, open(Path(args.out) / "results_gqa.json", "w"), indent=2)
        print(f"  acc {m['acc_full']:.4f}  acc@cert {m['acc_eps']:.4f}  "
              f"hops {m['avg_steps']:.2f}  updates {m['avg_updates']:.2f}")


if __name__ == "__main__":
    main()
