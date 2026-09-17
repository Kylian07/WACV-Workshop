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
