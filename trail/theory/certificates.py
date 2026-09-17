"""Computable certificates for TRAIL's mirror-descent reasoning.

Everything in this module is a *measurable quantity*, not a learned one.  Each
function corresponds to a statement in the paper and is checked numerically in
``tests/test_theory.py``.

Notation
--------
p            belief over N visual atoms, a point on the simplex  Delta^{N-1}
V            (N, d) matrix of visual atom embeddings
s            (N,) unary compatibility scores produced by the controller
M            (N, N) *column-stochastic* relational transport operator
             (M @ p)_j = sum_i M_ji p_i  moves belief mass along a relation
beta         weight of the relational consistency term

Energy (Eq. 3 in the paper)

    E(p) = -<s, p> + (beta / 2) * || p - M p ||_2^2

The quadratic term is convex (its Hessian (I-M)^T (I-M) is PSD), so E is a
convex function on the simplex for every beta >= 0.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# energy, gradient
# --------------------------------------------------------------------------
def energy(p: torch.Tensor, s: torch.Tensor, M: torch.Tensor | None, beta: float) -> torch.Tensor:
    """E(p) for a batch of beliefs. p:(B,N) s:(B,N) M:(B,N,N)."""
    lin = -(s * p).sum(-1)
    if M is None or beta == 0.0:
        return lin
    r = p - torch.einsum("bij,bj->bi", M, p)
    return lin + 0.5 * beta * (r * r).sum(-1)


def energy_grad(p: torch.Tensor, s: torch.Tensor, M: torch.Tensor | None, beta: float) -> torch.Tensor:
    """grad_p E(p) = -s + beta (I - M)^T (I - M) p."""
    if M is None or beta == 0.0:
        return -s
    r = p - torch.einsum("bij,bj->bi", M, p)
    g_quad = r - torch.einsum("bji,bj->bi", M, r)  # (I - M)^T r
    return -s + beta * g_quad


# --------------------------------------------------------------------------
# Theorem A -- the halting certificate
# --------------------------------------------------------------------------
def frank_wolfe_gap(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """G(p) = max_{u in Delta} <g, p - u> = <g, p> - min_i g_i.

    For convex E this is a *sound upper bound on the true suboptimality*:

        E(p) - min_{u in Delta} E(u)  <=  G(p).

    Proof: convexity gives E(u) >= E(p) + <g, u - p> for every u, so
    E(p) - E(u) <= <g, p - u> <= G(p).  Taking u = argmin finishes it.

    Consequence (the halting rule of the paper): stopping the first time
    G(p_t) <= eps returns an eps-optimal belief, whatever t turned out to be.
    The rule needs one max and one dot product -- O(N) per step, no extra
    network, no learned gate.
    """
    return (g * p).sum(-1) - g.min(-1).values


# --------------------------------------------------------------------------
# Theorem B -- relative smoothness and monotone descent
# --------------------------------------------------------------------------
def relative_smoothness_constant(M: torch.Tensor | None, beta: float) -> float:
    """The paper's dimension-free modulus L = 4 * beta.

    Lemma 1.  Let H = beta (I - M)^T (I - M) be the Hessian of E and let M be
    column-stochastic.  Then

        ||H||_{1 -> inf} = max_ij |H_ij| <= 4 beta,

    *independently of the number of atoms N*.

    Proof.  H_ij = beta <(I-M) e_i, (I-M) e_j>.  Column i of M lies in the
    simplex, hence ||M e_i||_2 <= ||M e_i||_1 = 1 and
    ||(I-M) e_i||_2 <= ||e_i||_2 + ||M e_i||_2 <= 2.  Cauchy-Schwarz gives
    |H_ij| <= beta * 2 * 2 = 4 beta.  []

    Combined with Pinsker (||q - p||_1^2 <= 2 KL(q||p)) this yields the Bregman
    descent lemma with modulus L = 4 beta:

        E(q) <= E(p) + <grad E(p), q - p> + L * KL(q || p).

    So the *theoretically prescribed* mirror-descent step size is eta = 1 / (4
    beta): no step-size tuning, and it does not change when the feature grid is
    made finer.  This is the property that lets TRAIL keep iterating at test
    time on inputs whose N differs from training.
    """
    return 4.0 * float(beta)


def prescribed_step_size(beta: float, safety: float = 1.0) -> float:
    """eta = safety / (4 beta).  safety=1 gives monotonicity, 0.5 strict descent."""
    if beta <= 0:
        return 1.0  # linear energy: any eta works; 1.0 makes Prop. 1 exact
    return safety / (4.0 * float(beta))


def kl(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (q * ((q + eps).log() - (p + eps).log())).sum(-1)


def descent_residual(
    p_t: torch.Tensor,
    p_next: torch.Tensor,
    s: torch.Tensor,
    M: torch.Tensor | None,
    beta: float,
    eta: float,
) -> torch.Tensor:
    """Slack in the NoLips descent inequality; >= 0 means the step obeyed theory.

        E(p_{t+1}) <= E(p_t) - (1/eta - L) * KL(p_{t+1} || p_t),   L = 4 beta

    Returned value is  rhs - lhs.  With eta = 1/L the bound degenerates to plain
    monotonicity E(p_{t+1}) <= E(p_t).
    """
    L = relative_smoothness_constant(M, beta)
    e_t = energy(p_t, s, M, beta)
    e_n = energy(p_next, s, M, beta)
    slack_coeff = (1.0 / eta) - L
    return (e_t - slack_coeff * kl(p_next, p_t)) - e_n


# --------------------------------------------------------------------------
# Proposition 2 -- off-manifold drift
# --------------------------------------------------------------------------
def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row of v onto the probability simplex."""
    N = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(-1) - 1.0
    ind = torch.arange(1, N + 1, device=v.device, dtype=v.dtype)
    cond = u - css / ind > 0
    rho = cond.to(v.dtype).cumsum(-1).argmax(-1, keepdim=True)
    theta = css.gather(-1, rho) / (rho.to(v.dtype) + 1.0)
    return (v - theta).clamp_min(0.0)


def hull_drift(z: torch.Tensor, V: torch.Tensor, iters: int = 200) -> torch.Tensor:
    """dist(z, conv(V)) by FISTA on  min_{p in Delta} ||V^T p - z||^2.

    For TRAIL this is *identically zero* by construction (Prop. 2): the visual
    state is z_t = V^T p_t with p_t in the simplex.  For free-form latent
    recursion z_{t+1} = z_t + f(z_t, c_t) it is unconstrained, and what it does
    is Fig. 3.

    z: (B, d)   V: (B, N, d)  ->  (B,)
    """
    B, N, d = V.shape
    L = 2.0 * torch.linalg.matrix_norm(V, ord=2) ** 2          # (B,) grad-Lipschitz
    step = (1.0 / L.clamp_min(1e-12))[:, None]
    p = torch.full((B, N), 1.0 / N, device=V.device, dtype=V.dtype)
    y, t_k = p.clone(), 1.0
    for _ in range(iters):
        resid = torch.einsum("bn,bnd->bd", y, V) - z
        grad = 2.0 * torch.einsum("bnd,bd->bn", V, resid)
        p_new = project_simplex(y - step * grad)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5)
        y = p_new + ((t_k - 1.0) / t_next) * (p_new - p)
        p, t_k = p_new, t_next
    return (torch.einsum("bn,bnd->bd", p, V) - z).norm(dim=-1)


# --------------------------------------------------------------------------
# Proposition 3 -- margin-controlled halting time (beta = 0 case)
# --------------------------------------------------------------------------
def margin(s: torch.Tensor) -> torch.Tensor:
    """Delta = s_(1) - s_(2), the decision margin of the linear energy."""
    top2 = s.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def predicted_halting_time(s: torch.Tensor, eta: float, eps: float) -> torch.Tensor:
    """Prop. 3.  For beta = 0, MWU from the uniform belief gives in closed form

        p_t = softmax(t * eta * s),

    so the gap obeys  G_t = max_i s_i - <s, p_t> <= (N - 1) * S * exp(-t eta Delta)
    with S = max_i s_i - min_i s_i.  Solving for G_t <= eps:

        T(eps) <= ceil( log((N-1) S / eps) / (eta * Delta) ).

    The halting time is therefore *inversely proportional to the decision
    margin*: low-margin (hard, ambiguous) inputs are given more reasoning steps,
    with no supervision on the number of steps.  Section 5.3 tests this
    prediction against CLEVR's ground-truth program depth.
    """
    N = s.shape[-1]
    S = s.max(-1).values - s.min(-1).values
    d = margin(s).clamp_min(1e-6)
    return torch.ceil(torch.log(((N - 1) * S / eps).clamp_min(1.0 + 1e-6)) / (eta * d))
