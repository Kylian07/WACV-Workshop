"""CLEVR loading, splitting and feature caching, written for a Kaggle session.

Three things here are not boilerplate and matter for the paper:

1. ``program_depth`` extracts, per question, the length of CLEVR's ground-truth
   functional program and the number of *relational* nodes in it.  Neither is
   ever shown to the model; both are used in Sec. 5.3 to test whether the
   certificate-driven halting step discovers question difficulty on its own.

2. ``make_split`` builds the two generalisation splits the paper needs:
     depth  train on short programs, test on long ones -- the setting where a
            fixed-depth recurrence cannot help itself and an anytime one can;
     hoc    hold out an attribute *combination* (default red & cube) from
            training, test only on questions that mention it.

3. ``cache_features`` writes a float16 memmap of frozen-trunk atoms so the
   reasoning study fits in one Kaggle session.  Featurising is done once; every
   model in the comparison then trains off the same cache, which is also what
   makes the comparison fair.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_PUNCT = re.compile(r"([;?.,!])")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
CLEVR_MARKERS = ("questions/CLEVR_train_questions.json", "CLEVR_train_questions.json")


def find_clevr_root(hint: str | None = None) -> Path:
    """Locate CLEVR_v1.0 without hard-coding a Kaggle dataset slug.

    Kaggle mounts datasets at /kaggle/input/<slug>/..., and slugs differ between
    mirrors, so we search rather than assume.  Set ``Config.data_root`` to skip.
    """
    cands = []
    if hint:
        cands.append(Path(hint))
    for base in ("/kaggle/input", "/kaggle/working", "./data", os.environ.get("CLEVR_ROOT", "")):
        if base and Path(base).exists():
            cands.append(Path(base))
    for base in cands:
        if (base / "questions" / "CLEVR_train_questions.json").exists():
            return base
        for m in CLEVR_MARKERS:
            hits = sorted(base.glob(f"**/{m}"))
            if hits:
                root = hits[0].parent
                return root.parent if root.name == "questions" else root
    raise FileNotFoundError(
        "CLEVR not found. Add a CLEVR v1.0 dataset to the notebook (e.g. the Kaggle "
        "dataset 'timoboz/clevr-dataset'), or set Config.data_root to the directory "
        "that contains questions/ and images/."
    )


# --------------------------------------------------------------------------
# streaming question reader (CLEVR_train_questions.json is ~800 MB)
# --------------------------------------------------------------------------
def iter_questions(path: Path, limit: int = 0, keep_program: bool = True):
    """Yield question dicts one at a time without materialising the whole file."""
    dec = json.JSONDecoder()
    buf, n = "", 0
    with open(path, "r", encoding="utf-8") as f:
        # advance to the start of the "questions" array
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                return
            buf += chunk
            i = buf.find('"questions"')
            if i >= 0:
                j = buf.find("[", i)
                if j >= 0:
                    buf = buf[j + 1:]
                    break
        while True:
            buf = buf.lstrip()
            if buf[:1] == ",":
                buf = buf[1:].lstrip()
            if buf[:1] == "]":
                return
            if not buf:
                more = f.read(1 << 20)
                if not more:
                    return
                buf += more
                continue
            try:
                obj, end = dec.raw_decode(buf)
            except ValueError:
                more = f.read(1 << 20)
                if not more:
                    return
                buf += more
                continue
            buf = buf[end:]
            rec = {
                "question": obj["question"],
                "answer": obj.get("answer"),
                "image_index": obj["image_index"],
                "image_filename": obj.get("image_filename"),
            }
            if keep_program and "program" in obj:
                rec["program"] = [p["function"] if "function" in p else p.get("type")
                                  for p in obj["program"]]
            yield rec
            n += 1
            if limit and n >= limit:
                return


RELATIONAL = {"relate", "same_size", "same_color", "same_material", "same_shape"}


def program_depth(prog) -> tuple[int, int]:
    """(program length, number of relational nodes).  Both are ground truth only."""
    if not prog:
        return 0, 0
    return len(prog), sum(1 for f in prog if f in RELATIONAL)


def tokenize(s: str):
    return _PUNCT.sub(r" \1 ", s.lower()).split()


# --------------------------------------------------------------------------
def build_index(root: Path, cfg, split: str = "train"):
    """Read questions, apply the split rule, build vocabularies."""
    qpath = root / "questions" / f"CLEVR_{split}_questions.json"
    if not qpath.exists():
        qpath = next(root.glob(f"**/CLEVR_{split}_questions.json"))
    limit = cfg.max_train_questions if split == "train" else cfg.max_val_questions
    # over-read when a split filter will discard most questions
    read_limit = limit * 8 if (limit and cfg.split != "iid") else limit
    recs = []
    for r in iter_questions(qpath, limit=read_limit):
        d, nrel = program_depth(r.get("program"))
        r["depth"], r["n_rel"] = d, nrel
        recs.append(r)
    recs = apply_split(recs, cfg, split)
    if limit:
        recs = recs[:limit]
    return recs


def apply_split(recs, cfg, split):
    if cfg.split == "iid":
        return recs
    if cfg.split == "depth":
        if split == "train":
            return [r for r in recs if r["depth"] <= cfg.depth_train_max]
        return [r for r in recs if r["depth"] >= cfg.depth_test_min]
    if cfg.split == "hoc":
        a, b = cfg.hoc_heldout

        def mentions(r):
            q = r["question"].lower()
            return (a in q) and (b in q)

        if split == "train":
            return [r for r in recs if not mentions(r)]
        return [r for r in recs if mentions(r)]
    raise ValueError(cfg.split)


def build_vocab(recs, min_count: int = 1):
    wc = Counter()
    for r in recs:
        wc.update(tokenize(r["question"]))
    words = ["<pad>", "<unk>"] + sorted(w for w, c in wc.items() if c >= min_count)
    stoi = {w: i for i, w in enumerate(words)}
    answers = sorted({r["answer"] for r in recs if r["answer"] is not None})
    atoi = {a: i for i, a in enumerate(answers)}
    return stoi, atoi


# --------------------------------------------------------------------------
# feature cache
# --------------------------------------------------------------------------
def cache_features(root: Path, cfg, image_indices, split: str, device="cuda"):
    """CLEVR images are CLEVR_<split>_%06d.png under images/<split>/."""
    from .features import cache_features as _cache

    img_dir = root / "images" / split
    if not img_dir.exists():
        hits = sorted(root.glob(f"**/images/{split}"))
        if hits:
            img_dir = hits[0]
    return _cache(cfg, image_indices,
                  lambda ix: img_dir / f"CLEVR_{split}_{int(ix):06d}.png",
                  tag=split, device=device)


# --------------------------------------------------------------------------
class ClevrFeatures(Dataset):
    """Questions + cached atoms.  Carries depth labels for the analysis only."""

    def __init__(self, recs, feats, img_pos, stoi, atoi, max_q_len=46):
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
