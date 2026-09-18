"""Read-out heads.

The visual state z_T is a convex combination of atoms, which is exactly what
makes Prop. 2 true -- and exactly what makes cardinality hard to read off, since
convex combinations are scale-free.  We therefore give the head three views of
the terminal state:

  z_T   the on-manifold visual state            (what / where)
  p_T   the belief itself, via its entropy      (how concentrated)
  s_T   the *unnormalised* scores               (how many, via soft cardinality)

The soft-cardinality term  n_hat = sum_i sigmoid(a s_i + b)  is the only place
scale information enters, and it is interpretable on its own: Sec. 5.4 plots it
against CLEVR's ground-truth object counts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AnswerHead(nn.Module):
    def __init__(self, d_vis: int, d_ctrl: int, n_answers: int, hidden: int = 512,
                 use_count: bool = True, dropout: float = 0.15):
        super().__init__()
        self.use_count = use_count
        extra = 2 + (1 if use_count else 0)
        self.net = nn.Sequential(
            nn.Linear(d_vis + 2 * d_ctrl + extra, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_answers),
        )
        if use_count:
            self.count_a = nn.Parameter(torch.tensor(1.0))
            self.count_b = nn.Parameter(torch.tensor(0.0))

    def soft_count(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.count_a * s + self.count_b).sum(-1, keepdim=True)

    def forward(self, z, c, q, p, s):
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        ent = ent / math.log(p.shape[-1])                     # in [0, 1]
        feats = [z, c, q, ent, p.max(-1, keepdim=True).values]
        if self.use_count:
            feats.append(self.soft_count(s).log1p())
        return self.net(torch.cat(feats, dim=-1))
