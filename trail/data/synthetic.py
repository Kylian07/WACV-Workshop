"""HOPWORLD: a controlled k-hop visual reasoning task.

Why it is in the paper.  CLEVR tells us *how many* program nodes a question has
but not which reasoning step consumed which node, and its images take hours to
featurise.  HOPWORLD gives exact control over the one variable the paper is
about -- the number of reasoning hops a question requires -- at a scale that
runs on a CPU in minutes.  It is where Prop. 3 (halting time tracks difficulty)
and the depth-extrapolation claim are tested without confounds, before the same
claims are re-tested on CLEVR.

Task.  A G x G grid holds K objects, each with a colour and a shape.  A question
names a starting object by colour, then gives k direction hops; each hop moves
to the nearest object strictly in that direction (staying put if there is none).
The answer is the colour or the shape of the object reached.

    "start red left left what shape"          ->  k = 2

Everything a hop needs is visible, so the task is solvable; nothing about k is
told to the model, so halting behaviour is free to be wrong.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

COLORS = ["red", "green", "blue", "yellow", "purple", "cyan", "gray", "brown"]
SHAPES = ["cube", "sphere", "cylinder", "cone"]
DIRS = ["left", "right", "above", "below"]
QUERY = ["color", "shape"]

VOCAB = ["<pad>", "start", "what"] + COLORS + SHAPES + DIRS + QUERY
STOI = {w: i for i, w in enumerate(VOCAB)}
ANSWERS = COLORS + SHAPES
ATOI = {a: i for i, a in enumerate(ANSWERS)}


def _step(pos, occupied, d):
    """Nearest occupied cell strictly in direction d, else stay."""
    y, x = pos
    cand = []
    for (yy, xx) in occupied:
        if d == "left" and xx < x and yy == y:
            cand.append((x - xx, (yy, xx)))
        elif d == "right" and xx > x and yy == y:
            cand.append((xx - x, (yy, xx)))
        elif d == "above" and yy < y and xx == x:
            cand.append((y - yy, (yy, xx)))
        elif d == "below" and yy > y and xx == x:
            cand.append((yy - y, (yy, xx)))
    if not cand:
        return pos
    return min(cand)[1]


class HopWorld(Dataset):
    def __init__(self, n: int, grid: int = 7, n_objects: int = 8, hops=(1, 2, 3),
                 seed: int = 0, max_q_len: int = 12):
        self.grid, self.max_q_len = grid, max_q_len
        rng = np.random.RandomState(seed)
        self.items = []
        # objects live on rows/columns so that "nearest in direction" is well posed
        while len(self.items) < n:
            cells = [(y, x) for y in range(grid) for x in range(grid)]
            idx = rng.choice(len(cells), size=n_objects, replace=False)
            occupied = [cells[i] for i in idx]
            col = rng.randint(0, len(COLORS), size=n_objects)
            shp = rng.randint(0, len(SHAPES), size=n_objects)
            by_cell = {c: (COLORS[col[i]], SHAPES[shp[i]]) for i, c in enumerate(occupied)}
            # a unique colour to anchor the start
            uniq = [i for i in range(n_objects) if (col == col[i]).sum() == 1]
            if not uniq:
                continue
            si = int(rng.choice(uniq))
            k = int(rng.choice(hops))
            ds = [DIRS[i] for i in rng.randint(0, len(DIRS), size=k)]
            pos = occupied[si]
            for d in ds:
                pos = _step(pos, occupied, d)
            qa = QUERY[rng.randint(0, 2)]
            ans = by_cell[pos][0] if qa == "color" else by_cell[pos][1]
            tokens = ["start", COLORS[col[si]]] + ds + ["what", qa]
            self.items.append(dict(
                occupied=occupied, by_cell=by_cell, tokens=tokens, answer=ans,
                hops=k, target_cell=pos, start_cell=occupied[si],
            ))

    # -- feature map: one-hot colour x shape per cell ----------------------
    def _image(self, it):
        G = self.grid
        F = len(COLORS) + len(SHAPES) + 1
        img = np.zeros((G * G, F), dtype=np.float32)
        for (y, x), (c, s) in it["by_cell"].items():
            j = y * G + x
            img[j, COLORS.index(c)] = 1.0
            img[j, len(COLORS) + SHAPES.index(s)] = 1.0
            img[j, -1] = 1.0                      # occupancy
        return img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        q = [STOI[t] for t in it["tokens"]][: self.max_q_len]
        q = q + [0] * (self.max_q_len - len(q))
        gy, gx = it["target_cell"]
        return {
            "feat": torch.from_numpy(self._image(it)),
            "question": torch.tensor(q, dtype=torch.long),
            "answer": torch.tensor(ATOI[it["answer"]], dtype=torch.long),
            "hops": torch.tensor(it["hops"], dtype=torch.long),
            "target_atom": torch.tensor(gy * self.grid + gx, dtype=torch.long),
        }


def hopworld_splits(n_train=20000, n_val=4000, grid=7, n_objects=8,
                    train_hops=(1, 2, 3), test_hops=(1, 2, 3), seed=0):
    tr = HopWorld(n_train, grid, n_objects, train_hops, seed=seed)
    va = HopWorld(n_val, grid, n_objects, test_hops, seed=seed + 1)
    return tr, va, len(VOCAB), len(ANSWERS), len(COLORS) + len(SHAPES) + 1
