"""TRAIL: Transport-Regularised Anytime Iterated Latent reasoning.

One reasoning step is one entropic-mirror-descent step on a question-conditioned
energy defined over the simplex of *actual visual atoms*.  Three consequences,
all of them free:

  * on-manifold by construction   z_t = V^T p_t in conv(V)            (Prop. 2)
  * a computable halting certificate  G_t = <g_t, p_t> - min_i g_t,i  (Thm. A)
  * a decodable trace             p_t is literally a heat map          (Sec. 5.4)

and one that is not free but is prescribed rather than tuned: the step size
eta = 1/(4 beta) coming from Lemma 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..theory import certificates as cert


# --------------------------------------------------------------------------
@dataclass
class TrailState:
    """Everything the analysis scripts need; nothing the model needs to learn."""

    p: list = field(default_factory=list)         # belief after each hop   [H] x (B,N)
    gap: list = field(default_factory=list)       # per-hop FW gap          [H] x (B,)
    inner_gap: list = field(default_factory=list)  # every MD iteration     [H*K] x (B,)
    move: list = field(default_factory=list)      # KL(p_h || p_{h-1})      [H] x (B,)
    energy: list = field(default_factory=list)    # E_h(p_h)                [H] x (B,)
    logits: list = field(default_factory=list)    # anytime answers         [H] x (B,A)
    halt_step: torch.Tensor | None = None         # first hop meeting a certificate

    def stack(self):
        out = {}
        for k in ("p", "gap", "inner_gap", "move", "energy", "logits"):
            v = getattr(self, k)
            if v:
                out[k] = torch.stack(v, dim=1)   # (B, T, ...)
        if self.halt_step is not None:
            out["halt_step"] = self.halt_step
        return out


# --------------------------------------------------------------------------
class ControlUnit(nn.Module):
    """Reads the next sub-question off the question words (MAC-style).

    The control vector is a convex combination of *contextual word states*, so
    it is grounded in the question rather than being a free-floating vector.
    The recurrent part of TRAIL that is allowed to be unconstrained lives here
    and nowhere else -- the visual reasoning state stays on the simplex.
    """

    def __init__(self, d_ctrl: int, d_vis: int, n_steps: int):
        super().__init__()
        self.step_emb = nn.Parameter(torch.randn(n_steps, d_ctrl) * 0.02)
        self.mix = nn.Linear(2 * d_ctrl + d_vis, d_ctrl)
        self.score = nn.Linear(d_ctrl, 1)
        self.norm = nn.LayerNorm(d_ctrl)

    def forward(self, c_prev, q_vec, words, word_mask, z_t, t):
        # words: (B,L,d_ctrl)  word_mask: (B,L) bool, True = real token
        q_t = q_vec + self.step_emb[min(t, self.step_emb.shape[0] - 1)]
        u = self.mix(torch.cat([c_prev, q_t, z_t], dim=-1))
        att = self.score(torch.tanh(u.unsqueeze(1) * words))          # (B,L,1)
        att = att.masked_fill(~word_mask.unsqueeze(-1), -1e4)
        w = att.softmax(dim=1)
        return self.norm((w * words).sum(dim=1))


class RelationalTransport(nn.Module):
    """Builds the column-stochastic transport operator M(c) of Eq. 4.

    M_ji = P(mass at atom i moves to atom j | relation c).  Column-stochastic is
    the whole point: it is what makes Lemma 1 dimension-free, and it is what
    makes ``p -> M p`` a genuine redistribution of belief rather than a generic
    linear map.
    """

    def __init__(self, d_vis: int, d_ctrl: int, n_heads: int = 4, grid: int = 14,
                 use_pos_bias: bool = True):
        super().__init__()
        self.h = n_heads
        self.dk = d_vis // n_heads
        self.q = nn.Linear(d_vis + d_ctrl, d_vis)
        self.k = nn.Linear(d_vis + d_ctrl, d_vis)
        self.use_pos_bias = use_pos_bias
        if use_pos_bias:
            # relation-conditioned relative-position bias: lets "left of" /
            # "behind" become an actual transport direction on the grid.
            self.pos = nn.Sequential(nn.Linear(d_ctrl, 64), nn.ReLU(), nn.Linear(64, 9 * n_heads))
            self.register_buffer("rel", _relative_offsets(grid), persistent=False)

    def forward(self, V, c):
        B, N, d = V.shape
        cc = c.unsqueeze(1).expand(B, N, -1)
        qv = self.q(torch.cat([V, cc], -1)).view(B, N, self.h, self.dk).transpose(1, 2)
        kv = self.k(torch.cat([V, cc], -1)).view(B, N, self.h, self.dk).transpose(1, 2)
        logits = torch.einsum("bhjd,bhid->bhji", qv, kv) / (self.dk ** 0.5)   # (B,h,j,i)
        if self.use_pos_bias and self.rel.shape[0] == N:
            basis = self.pos(c).view(B, self.h, 9)                            # (B,h,9)
            logits = logits + torch.einsum("bhk,jik->bhji", basis, self.rel)
        M = logits.softmax(dim=2).mean(dim=1)      # softmax over destination j
        return M                                    # (B,N,N), columns sum to 1


def _relative_offsets(grid: int) -> torch.Tensor:
    """(N, N, 9) one-hot-ish basis over sign(dy) x sign(dx) for an HxW grid."""
    idx = torch.arange(grid * grid)
    y, x = idx // grid, idx % grid
    dy = (y[:, None] - y[None, :]).sign() + 1        # in {0,1,2}
    dx = (x[:, None] - x[None, :]).sign() + 1
    code = dy * 3 + dx                               # in {0..8}
    return F.one_hot(code, 9).float()


# --------------------------------------------------------------------------
class TrailReasoner(nn.Module):
    """The reasoning module.  Drop-in replacement for a stack of cross-attentions.

    Setting ``n_steps=1, beta=0, learn_eta=False`` makes it *exactly* softmax
    cross-attention (Prop. 1) -- which is why it can be initialised from, and
    compared against, an attention baseline at matched parameter count.
    """

    def __init__(
        self,
        d_vis: int,
        d_ctrl: int,
        n_steps: int = 4,
        inner_steps: int = 4,
        beta: float = 1.0,
        grid: int = 14,
        n_heads: int = 4,
        learn_eta: bool = False,
        eta_safety: float = 1.0,
        use_transport: bool = True,
        use_pos_bias: bool = True,
        temp: float = 1.0,
        delta: float = 1e-4,
        gap_mode: str = "relative",
    ):
        super().__init__()
        self.produces_certificate = True   # see evaluate() in train.py
        self.n_steps, self.beta, self.temp = n_steps, beta, temp
        self.inner_steps, self.delta, self.gap_mode = inner_steps, delta, gap_mode
        self.use_transport = use_transport and beta > 0
        self.control = ControlUnit(d_ctrl, d_vis, n_steps)
        self.to_query = nn.Linear(d_ctrl, d_vis)
        self.transport = (
            RelationalTransport(d_vis, d_ctrl, n_heads, grid, use_pos_bias)
            if self.use_transport else None
        )
        eta0 = cert.prescribed_step_size(beta if self.use_transport else 0.0, eta_safety)
        if learn_eta:
            self.log_eta = nn.Parameter(torch.tensor(float(eta0)).log())
        else:
            self.register_buffer("log_eta", torch.tensor(float(eta0)).log(), persistent=False)

    @property
    def eta(self) -> torch.Tensor:
        return self.log_eta.exp()

    def unary(self, V, c):
        """s(c)_i = <V_i, W c> / sqrt(d)."""
        return torch.einsum("bnd,bd->bn", V, self.to_query(c)) / (V.shape[-1] ** 0.5) / self.temp

    def forward(
        self,
        V: torch.Tensor,            # (B,N,d) visual atoms
        q_vec: torch.Tensor,        # (B,d_ctrl)
        words: torch.Tensor,        # (B,L,d_ctrl)
        word_mask: torch.Tensor,    # (B,L) bool
        n_steps: int | None = None,
        eps: float | None = None,
        record: bool = True,
        answer_head=None,
    ) -> TrailState:
        """Two loops, because that is what the theorems are about.

        OUTER (h = 1..H): the controller reads the next sub-question off the
        question words and *fixes* an energy E_h(p) = -<s_h, p> + (beta/2)
        ||p - M_h p||^2.  One outer step = one hop.

        INNER (k = 1..K): entropic mirror descent on that fixed E_h, warm-started
        from the previous hop's belief.  Because E_h does not move during the
        inner loop, Lemma 1, Thm. A and Thm. B apply verbatim: the step size is
        the prescribed 1/(4 beta), the energy is non-increasing, and the
        Frank-Wolfe gap is a sound bound on how far this hop is from solved.

        Two halting rules, both certificates rather than learned gates:
          inner   G_k <= eps_rel * G_0   -- this hop is solved to eps_rel
          outer   KL(p_h || p_{h-1}) <= delta -- the hop changed nothing, so
                  there is no further sub-question to answer.

        Cost note: the expensive object is M_h, built once per *hop*, not once
        per mirror-descent step.  H=4, K=4 costs 4 transport builds and 16 belief
        updates -- cheaper than 8 flat steps, which would cost 8 of each.
        """
        B, N, _ = V.shape
        H = n_steps or self.n_steps
        K = self.inner_steps
        eta = self.eta
        log_p = torch.full((B, N), -math.log(N), device=V.device, dtype=V.dtype)
        p = log_p.exp()
        c = torch.zeros(B, q_vec.shape[-1], device=V.device, dtype=V.dtype)
        beta = self.beta if self.use_transport else 0.0
        st = TrailState()
        done = torch.zeros(B, dtype=torch.bool, device=V.device)
        halt = torch.full((B,), float(H), device=V.device)
        gap0 = None

        for h in range(H):
            z = torch.einsum("bn,bnd->bd", p, V)
            c = self.control(c, q_vec, words, word_mask, z, h)
            s = self.unary(V, c)
            M = self.transport(V, c) if self.use_transport else None
            p_hop_start = p

            inner_gaps = []
            for k in range(K):
                g = cert.energy_grad(p, s, M, beta)
                gap = cert.frank_wolfe_gap(p, g)
                inner_gaps.append(gap)
                if record:
                    st.inner_gap.append(gap)
                log_p = torch.log_softmax(log_p - eta * g, dim=-1)
                p = log_p.exp()

            g_end = cert.energy_grad(p, s, M, beta)
            gap_end = cert.frank_wolfe_gap(p, g_end)
            move = cert.kl(p, p_hop_start)          # how much this hop moved belief

            z = torch.einsum("bn,bnd->bd", p, V)
            if gap0 is None:
                gap0 = inner_gaps[0].detach().clamp_min(1e-8)

            if record:
                st.p.append(p)
                st.gap.append(gap_end if self.gap_mode == "absolute" else gap_end / gap0)
                st.move.append(move)
                st.energy.append(cert.energy(p, s, M, beta))
                if answer_head is not None:
                    st.logits.append(answer_head(z, c, q_vec, p, s))

            if eps is not None:
                crit = gap_end if self.gap_mode == "absolute" else gap_end / gap0
                newly = (~done) & ((crit <= eps) | (move <= self.delta))
                halt = torch.where(newly, torch.full_like(halt, float(h)), halt)
                done = done | newly

        st.halt_step = halt
        return st
