"""Baseline reasoners, all sharing TRAIL's backbone, text encoder and head.

The point of the comparison is *not* to reproduce published CLEVR numbers
(those come from multi-day runs on full-resolution ResNet-101 features); it is
to hold everything except the reasoning module fixed and vary only how the
model iterates.  Every baseline below therefore exposes the same
``forward(V, q_vec, words, word_mask, ..., answer_head) -> TrailState``
interface, so ``train.py`` and every figure script are agnostic to which one is
running.

  attn1           a single cross-attention.  By Prop. 1 this is TRAIL at T=1,
                  beta=0, eta=1 -- the control that isolates *iteration* from
                  *architecture*.
  latent_freeform the current latent-visual-reasoning recipe: an unconstrained
                  residual recursion z <- z + f(z, c) in R^d for a fixed number
                  of steps.  Same parameter budget, same controller.  This is
                  the baseline Prop. 2 and Fig. 3 are about.
  mac             a compact MAC cell (control / read / write), fixed steps.
  film            FiLM-conditioned residual blocks over the feature map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trail_cell import ControlUnit, TrailState


class Attn1Reasoner(nn.Module):
    """TRAIL with the iteration removed."""

    def __init__(self, d_vis, d_ctrl, **kw):
        super().__init__()
        self.control = ControlUnit(d_ctrl, d_vis, 1)
        self.to_query = nn.Linear(d_ctrl, d_vis)

    def forward(self, V, q_vec, words, word_mask, n_steps=None, eps=None, record=True, answer_head=None):
        B, N, d = V.shape
        z0 = V.mean(1)
        c = self.control(torch.zeros_like(q_vec), q_vec, words, word_mask, z0, 0)
        s = torch.einsum("bnd,bd->bn", V, self.to_query(c)) / (d ** 0.5)
        p = s.softmax(-1)
        z = torch.einsum("bn,bnd->bd", p, V)
        st = TrailState()
        st.p = [p]
        st.gap = [torch.zeros(B, device=V.device)]
        if answer_head is not None:
            st.logits = [answer_head(z, c, q_vec, p, s)]
        st.halt_step = torch.zeros(B, device=V.device)
        return st


class FreeFormLatentReasoner(nn.Module):
    """z_{t+1} = z_t + f(z_t, c_t):  latent reasoning with no manifold constraint.

    Kept deliberately close to TRAIL -- same control unit, same width, a
    comparable parameter count -- so that differences in Fig. 3 (drift) and
    Fig. 5 (depth extrapolation) are attributable to the *constraint*, not to
    capacity.
    """

    def __init__(self, d_vis, d_ctrl, n_steps=8, hidden=512, share_steps=False, **kw):
        super().__init__()
        self.n_steps = n_steps
        self.control = ControlUnit(d_ctrl, d_vis, n_steps, share_steps)
        self.read = nn.Linear(d_ctrl, d_vis)
        self.upd = nn.Sequential(nn.Linear(2 * d_vis + d_ctrl, hidden), nn.ELU(),
                                 nn.Linear(hidden, d_vis))
        self.norm = nn.LayerNorm(d_vis)

    def forward(self, V, q_vec, words, word_mask, n_steps=None, eps=None, record=True, answer_head=None):
        B, N, d = V.shape
        T = n_steps or self.n_steps
        z = V.mean(1)
        c = torch.zeros_like(q_vec)
        st = TrailState()
        for t in range(T):
            c = self.control(c, q_vec, words, word_mask, z, t)
            s = torch.einsum("bnd,bd->bn", V, self.read(c)) / (d ** 0.5)
            ctx = torch.einsum("bn,bnd->bd", s.softmax(-1), V)
            z = self.norm(z + self.upd(torch.cat([z, ctx, c], -1)))
            if record:
                st.p.append(s.softmax(-1))
                st.gap.append(torch.zeros(B, device=V.device))
                if answer_head is not None:
                    st.logits.append(answer_head(z, c, q_vec, s.softmax(-1), s))
        st.halt_step = torch.full((B,), float(T), device=V.device)
        st.z_final = z          # consumed by the drift measurement in Fig. 3
        return st


class MACReasoner(nn.Module):
    """Compact MAC cell: control, read (attention over atoms), write (gated memory)."""

    def __init__(self, d_vis, d_ctrl, n_steps=12, share_steps=False, **kw):
        super().__init__()
        self.n_steps = n_steps
        self.control = ControlUnit(d_ctrl, d_vis, n_steps, share_steps)
        self.mem_proj = nn.Linear(d_vis, d_vis)
        self.read_proj = nn.Linear(2 * d_vis, d_vis)
        self.attn = nn.Linear(d_vis, 1)
        self.q_read = nn.Linear(d_ctrl, d_vis)
        self.write = nn.Linear(2 * d_vis, d_vis)
        self.gate = nn.Linear(d_ctrl, 1)
        self.norm = nn.LayerNorm(d_vis)

    def forward(self, V, q_vec, words, word_mask, n_steps=None, eps=None, record=True, answer_head=None):
        B, N, d = V.shape
        T = n_steps or self.n_steps
        m = V.mean(1)
        c = torch.zeros_like(q_vec)
        st = TrailState()
        for t in range(T):
            c = self.control(c, q_vec, words, word_mask, m, t)
            inter = self.mem_proj(m).unsqueeze(1) * V
            feat = self.read_proj(torch.cat([inter, V], -1)) * self.q_read(c).unsqueeze(1)
            a = self.attn(torch.tanh(feat)).softmax(1)
            r = (a * V).sum(1)
            g = torch.sigmoid(self.gate(c))
            m = self.norm(g * self.write(torch.cat([m, r], -1)) + (1 - g) * m)
            if record:
                st.p.append(a.squeeze(-1))
                st.gap.append(torch.zeros(B, device=V.device))
                if answer_head is not None:
                    st.logits.append(answer_head(m, c, q_vec, a.squeeze(-1), a.squeeze(-1)))
        st.halt_step = torch.full((B,), float(T), device=V.device)
        st.z_final = m
        return st


class FiLMReasoner(nn.Module):
    """FiLM residual blocks over the atom grid (treated as a sqrt(N) x sqrt(N) map)."""

    def __init__(self, d_vis, d_ctrl, n_steps=4, grid=14, **kw):
        super().__init__()
        self.grid, self.n_blocks = grid, n_steps
        self.film = nn.Linear(d_ctrl, 2 * d_vis * n_steps)
        self.c1 = nn.ModuleList([nn.Conv2d(d_vis, d_vis, 1) for _ in range(n_steps)])
        self.c2 = nn.ModuleList([nn.Conv2d(d_vis, d_vis, 3, padding=1) for _ in range(n_steps)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(d_vis, affine=False) for _ in range(n_steps)])
        self.head_attn = nn.Linear(d_vis, 1)

    def forward(self, V, q_vec, words, word_mask, n_steps=None, eps=None, record=True, answer_head=None):
        B, N, d = V.shape
        g = int(round(N ** 0.5))
        x = V.transpose(1, 2).reshape(B, d, g, g)
        gb = self.film(q_vec).view(B, self.n_blocks, 2, d)
        st = TrailState()
        for i in range(self.n_blocks):
            r = F.relu(self.c1[i](x))
            h = self.bn[i](self.c2[i](r))
            gamma, beta = gb[:, i, 0][:, :, None, None], gb[:, i, 1][:, :, None, None]
            x = r + F.relu((1 + gamma) * h + beta)
        feat = x.flatten(2).transpose(1, 2)
        a = self.head_attn(feat).softmax(1)
        z = (a * feat).sum(1)
        st.p = [a.squeeze(-1)]
        st.gap = [torch.zeros(B, device=V.device)]
        if answer_head is not None:
            st.logits = [answer_head(z, q_vec, q_vec, a.squeeze(-1), a.squeeze(-1))]
        st.halt_step = torch.zeros(B, device=V.device)
        st.z_final = z
        return st


REASONERS = {
    "attn1": Attn1Reasoner,
    "latent_freeform": FreeFormLatentReasoner,
    "mac": MACReasoner,
    "film": FiLMReasoner,
}
