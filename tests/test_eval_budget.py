"""Guards the thing that makes the comparison a comparison.

Every model in the paper is supposed to get the same belief-update budget: TRAIL
runs ``n_steps`` hops of ``inner_steps`` mirror-descent iterations, and the
recurrent baselines are built with ``n_steps * inner_steps`` iterations to match.

An earlier version of ``evaluate`` defaulted its depth to ``cfg.n_steps``, which
is TRAIL's *hop* count.  The baselines were therefore trained at 12 iterations and
evaluated at 4, which cost them accuracy and flattered TRAIL by about 20 points.
Nothing crashed and no number looked absurd, which is exactly why this is a test
rather than a comment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trail.config import Config
from trail.data.synthetic import hopworld_splits
from trail.models.model import VQAModel
from trail.train import evaluate

EXPECTED = {"trail": 4, "latent_freeform": 12, "mac": 12, "film": 1, "attn1": 1}


def test_every_reasoner_is_evaluated_at_its_built_depth():
    _, va, vocab, n_ans, fd = hopworld_splits(50, 128, 7, 8, seed=0)
    for name, expect in EXPECTED.items():
        cfg = Config(dataset="synthetic", reasoner=name, grid=7, d_vis=64, d_ctrl=64,
                     d_hidden=64, n_steps=4, inner_steps=3, batch_size=64,
                     num_workers=0, use_count=False, amp=False)
        model = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=fd)
        e = evaluate(model, va, cfg, device="cpu")
        got = len(e["gap_curve"]) if name == "trail" else round(e["avg_steps"])
        assert got == expect, f"{name}: evaluated at {got} steps, built with {expect}"
    print("every reasoner evaluated at its built depth: OK")


def test_budgets_are_matched_in_belief_updates():
    """TRAIL's hops x inner steps must equal the baselines' iteration count."""
    _, va, vocab, n_ans, fd = hopworld_splits(50, 128, 7, 8, seed=0)
    budgets = {}
    for name in ("trail", "latent_freeform", "mac"):
        cfg = Config(dataset="synthetic", reasoner=name, grid=7, d_vis=64, d_ctrl=64,
                     d_hidden=64, n_steps=4, inner_steps=3, batch_size=64,
                     num_workers=0, use_count=False, amp=False, delta=0.0)
        model = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=fd)
        budgets[name] = evaluate(model, va, cfg, device="cpu")["avg_updates"]
    assert len(set(round(v) for v in budgets.values())) == 1, budgets
    print(f"budgets matched at full depth: {budgets}")



def test_operating_point_is_always_measured():
    """A cfg.delta outside eps_sweep must still be measured, not silently faked.

    evaluate() used to fall back to `acc_full` and `float(T) * per_step` when
    cfg.delta was not one of the swept thresholds.  That produced a row reporting
    the architectural budget and full-depth accuracy -- plausible-looking numbers
    that describe nothing the model did.  It went unnoticed through a whole
    ablation run.
    """
    _, va, vocab, n_ans, fd = hopworld_splits(50, 256, 7, 8, seed=0)
    off_sweep = 0.77
    cfg = Config(dataset="synthetic", reasoner="trail", grid=7, d_vis=64, d_ctrl=64,
                 d_hidden=64, n_steps=4, inner_steps=3, batch_size=64, num_workers=0,
                 use_count=False, amp=False, delta=off_sweep)
    assert off_sweep not in cfg.eps_sweep, "pick a delta outside the sweep for this test"
    model = VQAModel(cfg, vocab=vocab, n_answers=n_ans, feat_dim=fd)
    e = evaluate(model, va, cfg, device="cpu")

    budget = cfg.n_steps * cfg.inner_steps
    assert any(d["delta"] == off_sweep for d in e["frontier"]), \
        "the configured operating point must appear in the frontier"
    assert e["avg_updates"] < budget, \
        f"avg_updates {e['avg_updates']} equals the architectural budget -- fallback fired"
    assert e["avg_steps"] <= cfg.n_steps
    print(f"off-sweep delta={off_sweep} measured: {e['avg_steps']:.2f} hops, "
          f"{e['avg_updates']:.2f} updates (budget {budget}): OK")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL:
        fn()
    print(f"\nAll {len(ALL)} budget checks passed.")