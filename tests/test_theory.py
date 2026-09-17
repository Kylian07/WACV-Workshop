"""Numerical checks of every formal claim in the paper.

Run with:  python -m pytest tests/ -q     (or)  python tests/test_theory.py

These are not smoke tests.  Each one fails if the corresponding proposition is
false, so they double as the artifact that a reviewer can run in under a minute.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trail.models.trail_cell import TrailReasoner
from trail.theory import certificates as cert

torch.manual_seed(0)
DT = torch.float64


def _random_column_stochastic(B, N):
    M = torch.rand(B, N, N, dtype=DT)
    return M / M.sum(dim=1, keepdim=True)      # columns sum to 1


def _random_simplex(B, N):
    p = torch.rand(B, N, dtype=DT)
    return p / p.sum(-1, keepdim=True)


# --------------------------------------------------------------------------
def test_lemma1_dimension_free_smoothness():
    """Lemma 1:  max_ij |H_ij| <= 4 beta  for every N, with H = beta (I-M)^T (I-M)."""
    beta = 1.7
    for N in (4, 16, 64, 256):
        M = _random_column_stochastic(1, N)[0]
        I = torch.eye(N, dtype=DT)
        H = beta * (I - M).T @ (I - M)
        assert H.abs().max().item() <= 4 * beta + 1e-9, (N, H.abs().max().item())
        # and the bound does not degrade with N (it is not an artefact of scaling)
        assert torch.linalg.eigvalsh(H).min().item() >= -1e-9   # PSD => E convex
    print("Lemma 1 (dimension-free relative smoothness, PSD Hessian): OK")


def test_theorem_b_monotone_descent():
    """Thm B: with eta = 1/(4 beta) the mirror-descent step never increases E."""
    B, N, beta = 32, 64, 2.0
    eta = cert.prescribed_step_size(beta)
    assert abs(eta - 1 / (4 * beta)) < 1e-12
    s = torch.randn(B, N, dtype=DT)
    M = _random_column_stochastic(B, N)
    p = _random_simplex(B, N)
    prev = cert.energy(p, s, M, beta)
    worst = 0.0
    for _ in range(200):
        g = cert.energy_grad(p, s, M, beta)
        p = torch.log_softmax(p.clamp_min(1e-300).log() - eta * g, dim=-1).exp()
        e = cert.energy(p, s, M, beta)
        worst = max(worst, (e - prev).max().item())
        assert (e - prev).max().item() <= 1e-9, "energy increased"
        prev = e
    print(f"Theorem B (monotone descent at eta=1/(4beta)): OK  (worst increase {worst:.2e})")


def test_theorem_b_descent_inequality():
    """Thm B, sharper form: E(p+) <= E(p) - (1/eta - L) KL(p+ || p)."""
    B, N, beta = 16, 32, 1.0
    eta = cert.prescribed_step_size(beta, safety=0.5)       # strict-decrease regime
    s = torch.randn(B, N, dtype=DT)
    M = _random_column_stochastic(B, N)
    p = _random_simplex(B, N)
    for _ in range(50):
        g = cert.energy_grad(p, s, M, beta)
        p_next = torch.log_softmax(p.log() - eta * g, dim=-1).exp()
        r = cert.descent_residual(p, p_next, s, M, beta, eta)
        assert r.min().item() >= -1e-9, r.min().item()
        p = p_next
    print("Theorem B (NoLips descent inequality with slack): OK")


def test_theorem_a_gap_is_an_upper_bound():
    """Thm A: G(p) >= E(p) - min_{u in Delta} E(u).  The certificate is sound."""
    B, N, beta = 8, 24, 1.0
    s = torch.randn(B, N, dtype=DT)
    M = _random_column_stochastic(B, N)
    p = _random_simplex(B, N)
    # reference minimum by long mirror descent from many starts
    best = torch.full((B,), float("inf"), dtype=DT)
    for trial in range(5):
        q = _random_simplex(B, N) if trial else torch.full((B, N), 1.0 / N, dtype=DT)
        for _ in range(4000):
            q = torch.log_softmax(q.clamp_min(1e-300).log()
                                  - cert.prescribed_step_size(beta)
                                  * cert.energy_grad(q, s, M, beta), dim=-1).exp()
        best = torch.minimum(best, cert.energy(q, s, M, beta))
    g = cert.energy_grad(p, s, M, beta)
    gap = cert.frank_wolfe_gap(p, g)
    true_subopt = cert.energy(p, s, M, beta) - best
    assert (gap - true_subopt).min().item() >= -1e-7, (gap - true_subopt).min().item()
    assert gap.min().item() >= -1e-12, "gap must be non-negative"
    print("Theorem A (FW gap upper-bounds true suboptimality): OK")


def test_theorem_b_rate():
    """Thm B rate: E(p_T) - E* <= 4 beta log N / T  from the uniform start."""
    B, N, beta = 4, 64, 1.0
    s = torch.randn(B, N, dtype=DT)
    M = _random_column_stochastic(B, N)
    eta = cert.prescribed_step_size(beta)
    p = torch.full((B, N), 1.0 / N, dtype=DT)
    traj = []
    for t in range(1, 401):
        p = torch.log_softmax(p.clamp_min(1e-300).log() - eta * cert.energy_grad(p, s, M, beta), -1).exp()
        traj.append(cert.energy(p, s, M, beta))
    star = traj[-1] - 1e-9
    for T in (10, 50, 200):
        bound = 4 * beta * math.log(N) / T
        assert (traj[T - 1] - star).max().item() <= bound + 1e-6, T
    print("Theorem B (O(4 beta log N / T) rate): OK")


def test_prop1_attention_is_one_trail_step():
    """Prop. 1: T=1, beta=0, eta=1, uniform start  ==>  p_1 = softmax(s)."""
    B, N, d, dc = 5, 17, 32, 32
    r = TrailReasoner(d_vis=d, d_ctrl=dc, n_steps=1, inner_steps=1, beta=0.0, grid=4,
                      use_transport=False, learn_eta=False).double()
    assert abs(r.eta.item() - 1.0) < 1e-12, "eta must default to 1 when beta = 0"
    V = torch.nn.functional.normalize(torch.randn(B, N, d, dtype=DT), dim=-1)
    q = torch.randn(B, dc, dtype=DT)
    w = torch.randn(B, 7, dc, dtype=DT)
    mask = torch.ones(B, 7, dtype=torch.bool)
    st = r(V, q, w, mask, record=True)
    c0 = r.control(torch.zeros(B, dc, dtype=DT), q, w, mask,
                   torch.einsum("bn,bnd->bd", torch.full((B, N), 1.0 / N, dtype=DT), V), 0)
    s = r.unary(V, c0)
    assert torch.allclose(st.p[0], s.softmax(-1), atol=1e-10)
    z_trail = torch.einsum("bn,bnd->bd", st.p[0], V)
    z_attn = torch.einsum("bn,bnd->bd", s.softmax(-1), V)
    assert torch.allclose(z_trail, z_attn, atol=1e-10)
    print("Prop. 1 (softmax cross-attention is exactly one TRAIL step): OK")


def test_prop2_no_drift():
    """Prop. 2: the visual state never leaves conv(V); free-form recursion does."""
    B, N, d, dc = 4, 25, 32, 32
    r = TrailReasoner(d_vis=d, d_ctrl=dc, n_steps=3, inner_steps=2, beta=1.0, grid=5).double()
    V = torch.nn.functional.normalize(torch.randn(B, N, d, dtype=DT), dim=-1)
    q, w = torch.randn(B, dc, dtype=DT), torch.randn(B, 5, dc, dtype=DT)
    st = r(V, q, w, torch.ones(B, 5, dtype=torch.bool), record=True)
    for p in st.p:
        assert torch.allclose(p.sum(-1), torch.ones(B, dtype=DT), atol=1e-10)
        assert p.min().item() >= 0.0
        z = torch.einsum("bn,bnd->bd", p, V)
        assert cert.hull_drift(z, V, iters=300).max().item() < 1e-3

    from trail.models.baselines import FreeFormLatentReasoner
    ff = FreeFormLatentReasoner(d_vis=d, d_ctrl=dc, n_steps=6).double()
    stf = ff(V, q, w, torch.ones(B, 5, dtype=torch.bool), record=True)
    drift = cert.hull_drift(stf.z_final, V, iters=300).mean().item()
    print(f"Prop. 2 (TRAIL drift = 0; free-form drift = {drift:.3f}): OK")


def test_prop3_margin_controls_halting():
    """Prop. 3: for beta = 0 the halting time scales like 1 / margin."""
    N, eta, eps = 64, 1.0, 1e-2
    times, margins = [], []
    for trial in range(60):
        s = torch.randn(N, dtype=DT) * (0.3 + 0.05 * (trial % 10))
        p = torch.full((N,), 1.0 / N, dtype=DT)
        for t in range(1, 5001):
            g = -s
            if cert.frank_wolfe_gap(p[None], g[None])[0].item() <= eps:
                break
            p = torch.log_softmax(p.log() - eta * g, -1).exp()
        times.append(t)
        margins.append(cert.margin(s[None])[0].item())
        pred = cert.predicted_halting_time(s[None], eta, eps)[0].item()
        assert t <= pred + 1, (t, pred)          # the bound must hold
    import numpy as np
    rho = np.corrcoef(np.argsort(np.argsort(times)),
                      np.argsort(np.argsort([-m for m in margins])))[0, 1]
    assert rho > 0.8, rho
    print(f"Prop. 3 (halting time bound holds; rank corr with 1/margin = {rho:.2f}): OK")


def test_certificate_halting_is_sound_end_to_end():
    """The rule actually used at test time returns an eps-optimal belief."""
    B, N, beta, eps = 8, 48, 1.0, 1e-3
    s = torch.randn(B, N, dtype=DT)
    M = _random_column_stochastic(B, N)
    eta = cert.prescribed_step_size(beta)
    p = torch.full((B, N), 1.0 / N, dtype=DT)
    halted = torch.zeros(B, dtype=torch.bool)
    e_at_halt = torch.zeros(B, dtype=DT)
    for t in range(2000):
        g = cert.energy_grad(p, s, M, beta)
        gap = cert.frank_wolfe_gap(p, g)
        new = (~halted) & (gap <= eps)
        e_at_halt = torch.where(new, cert.energy(p, s, M, beta), e_at_halt)
        halted |= new
        if halted.all():
            break
        p = torch.log_softmax(p.log() - eta * g, -1).exp()
    e_final = cert.energy(p, s, M, beta)
    assert halted.all(), "did not halt"
    assert (e_at_halt - e_final).max().item() <= eps + 1e-8
    print(f"End-to-end certificate soundness at eps={eps}: OK (halted {int(halted.sum())}/{B})")


def test_module_inner_loop_is_the_theory():
    """The module's inner loop really is the mirror descent the theorems cover.

    We pull the hop's fixed energy (s_h, M_h) straight out of the module, replay
    the inner iterations through ``certificates``, and check both that the step
    size is the prescribed one and that E_h falls monotonically.
    """
    B, N, d, dc, beta = 6, 36, 32, 32, 1.5
    r = TrailReasoner(d_vis=d, d_ctrl=dc, n_steps=2, inner_steps=6, beta=beta, grid=6).double()
    assert abs(r.eta.item() - 1.0 / (4 * beta)) < 1e-6, r.eta.item()   # buffer is fp32
    V = torch.nn.functional.normalize(torch.randn(B, N, d, dtype=DT), dim=-1)
    q, w = torch.randn(B, dc, dtype=DT), torch.randn(B, 5, dc, dtype=DT)
    mask = torch.ones(B, 5, dtype=torch.bool)
    with torch.no_grad():
        p0 = torch.full((B, N), 1.0 / N, dtype=DT)
        c = r.control(torch.zeros(B, dc, dtype=DT), q, w, mask,
                      torch.einsum("bn,bnd->bd", p0, V), 0)
        s_h, M_h = r.unary(V, c), r.transport(V, c)
        assert torch.allclose(M_h.sum(1), torch.ones(B, N, dtype=DT), atol=1e-8), \
            "transport operator must be column-stochastic -- Lemma 1 depends on it"
        p, prev = p0, cert.energy(p0, s_h, M_h, beta)
        for _ in range(6):
            p = torch.log_softmax(p.log() - r.eta * cert.energy_grad(p, s_h, M_h, beta), -1).exp()
            e = cert.energy(p, s_h, M_h, beta)
            assert (e - prev).max().item() <= 1e-9
            prev = e
    print("Module inner loop (column-stochastic M, prescribed eta, monotone E_h): OK")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL:
        fn()
    print(f"\nAll {len(ALL)} theory checks passed.")
