"""GQA support, exposing the same interface as the CLEVR loader.

GQA gives real photographs and, in ``semantic``/``types``, a ground-truth
operation chain per question -- so the hop-count analysis of Sec. 5.3 transfers
verbatim: ``len(q['semantic'])`` plays the role of CLEVR's program length and
the number of ``relate`` operations plays the role of ``n_rel``.

As with CLEVR the root is discovered rather than hard-coded, because GQA mirrors
on Kaggle use different folder layouts.  Point ``Config.data_root`` at the
directory holding the question json files if discovery fails.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .clevr import tokenize


def find_gqa_root(hint: str | None = None) -> Path:
    cands = [Path(hint)] if hint else []
    for base in ("/kaggle/input", "./data", os.environ.get("GQA_ROOT", "")):
        if base and Path(base).exists():
            cands.append(Path(base))
    for base in cands:
        for pat in ("**/train_balanced_questions.json", "**/val_balanced_questions.json"):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0].parent
    raise FileNotFoundError(
        "GQA not found. Attach a GQA mirror to the notebook and/or set Config.data_root "
        "to the folder containing {train,val}_balanced_questions.json."
    )


RELATIONAL_OPS = ("relate", "verify rel", "choose rel")


def gqa_depth(q) -> tuple[int, int]:
    sem = q.get("semantic") or []
    n_rel = sum(1 for op in sem if any(op.get("operation", "").startswith(r) for r in RELATIONAL_OPS))
    return len(sem), n_rel


def load_gqa_questions(root: Path, split: str, limit: int = 0):
    path = root / f"{split}_balanced_questions.json"
    with open(path) as f:
        raw = json.load(f)
    recs = []
    for qid, q in raw.items():
        d, nrel = gqa_depth(q)
        recs.append({
            "question": q["question"], "answer": q.get("answer"),
            "image_index": q["imageId"], "depth": d, "n_rel": nrel,
        })
        if limit and len(recs) >= limit:
            break
    return recs


def find_gqa_images(root: Path, hint: str | None = None) -> Path:
    """GQA images are <imageId>.jpg in a flat folder, whose name varies by mirror."""
    for base in filter(None, [Path(hint) if hint else None, root, root.parent,
                              Path("/kaggle/input")]):
        if not Path(base).exists():
            continue
        for name in ("images", "gqa_images", "allImages", "raw_images"):
            d = Path(base) / name
            if d.is_dir() and any(d.glob("*.jpg")):
                return d
        hits = sorted(Path(base).glob("**/*.jpg"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        "GQA images not found. Attach the GQA images dataset and/or set Config.data_root.")


def cache_gqa_features(root: Path, cfg, image_ids, split: str, device="cuda"):
    from .features import cache_features as _cache

    img_dir = find_gqa_images(root, cfg.data_root)
    return _cache(cfg, image_ids, lambda ix: img_dir / f"{ix}.jpg",
                  tag=f"gqa_{split}", device=device)


def build_gqa_vocab(recs, min_count: int = 2):
    """Answers are open-vocabulary in GQA; keep the ones that actually recur."""
    from collections import Counter

    wc, ac = Counter(), Counter()
    for r in recs:
        wc.update(tokenize(r["question"]))
        if r["answer"] is not None:
            ac[r["answer"]] += 1
    words = ["<pad>", "<unk>"] + sorted(w for w, c in wc.items() if c >= min_count)
    answers = [a for a, c in ac.most_common() if c >= min_count]
    return {w: i for i, w in enumerate(words)}, {a: i for i, a in enumerate(answers)}


class GqaFeatures(Dataset):
    """Same tensor contract as ``ClevrFeatures`` so train.py needs no branches."""

    def __init__(self, recs, feats, img_pos, stoi, atoi, max_q_len=32):
        self.recs, self.feats, self.img_pos = recs, feats, img_pos
        self.stoi, self.atoi, self.max_q_len = stoi, atoi, max_q_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        toks = [self.stoi.get(w, 1) for w in tokenize(r["question"])][: self.max_q_len]
        toks = toks + [0] * (self.max_q_len - len(toks))
        f = np.asarray(self.feats[self.img_pos[str(r["image_index"])]], dtype=np.float32)
        return {
            "feat": torch.from_numpy(f),
            "question": torch.tensor(toks, dtype=torch.long),
            "answer": torch.tensor(self.atoi.get(r["answer"], 0), dtype=torch.long),
            "hops": torch.tensor(r.get("n_rel", 0), dtype=torch.long),
            "depth": torch.tensor(r.get("depth", 0), dtype=torch.long),
        }
