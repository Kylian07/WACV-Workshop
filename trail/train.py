"""Training and evaluation loop shared by every model in the paper.

The loss is the one the theory asks for.  TRAIL is meant to be *anytime*: the
belief after any number of steps is a valid answer to read off, which is what
makes a certificate-driven stopping rule legitimate.  We therefore supervise
every prefix,

    L = (1/T) * sum_t CE(head(p_t), y),

rather than only the final step.  ``anytime_loss='last'`` recovers the usual
train-the-last-step objective and is reported as an ablation, because it is the
thing that breaks the compute-accuracy frontier of Fig. 4.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .theory import certificates as cert


# --------------------------------------------------------------------------
def step_weights(T: int, mode: str, device) -> torch.Tensor:
    if mode == "uniform":
        w = torch.ones(T, device=device)
    elif mode == "last":
        w = torch.zeros(T, device=device)
        w[-1] = 1.0
    elif mode == "linear":
        w = torch.arange(1, T + 1, device=device, dtype=torch.float)
    else:
        raise ValueError(mode)
    return w / w.sum()


def anytime_loss(logits, y, mode="uniform"):
    """logits: (B, T, A)."""
    B, T, A = logits.shape
    w = step_weights(T, mode, logits.device)
    ce = F.cross_entropy(logits.reshape(B * T, A), y.repeat_interleave(T), reduction="none")
    return (ce.view(B, T) * w).sum(-1).mean()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
def train(model, train_ds, val_ds, cfg, device="cuda", log=print):
    set_seed(cfg.seed)
    model = model.to(device)
    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, cfg.epochs * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup)) *
        0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = bool(cfg.amp) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(cfg.out_dir, exist_ok=True)
    history, gstep, t0 = [], 0, time.time()
    for ep in range(cfg.epochs):
        model.train()
        run_loss, run_acc, seen = 0.0, 0.0, 0
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                st, _ = model(batch, record=True)
                logits = torch.stack(st.logits, dim=1)
                loss = anytime_loss(logits, batch["answer"], cfg.anytime_loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1
            bs = batch["answer"].shape[0]
            run_loss += loss.item() * bs
            run_acc += (logits[:, -1].argmax(-1) == batch["answer"]).float().sum().item()
            seen += bs
            if gstep % cfg.log_every == 0:
                log(f"  ep{ep} step{gstep} loss {run_loss/seen:.4f} acc {run_acc/seen:.4f} "
                    f"({time.time()-t0:.0f}s)")
        m = evaluate(model, val_ds, cfg, device=device)
        m.update(epoch=ep, train_loss=run_loss / seen, train_acc=run_acc / seen)
        history.append(m)
        log(f"[ep {ep}] val acc {m['acc_eps']:.4f} @ delta={cfg.delta} "
            f"(hops {m['avg_steps']:.2f}, updates {m['avg_updates']:.2f})  "
            f"acc_full {m['acc_full']:.4f}")
        torch.save({"model": model.state_dict(), "cfg": vars(cfg)},
                   Path(cfg.out_dir) / "last.pt")
        json.dump(history, open(Path(cfg.out_dir) / "history.json", "w"), indent=2)
    return history


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, ds, cfg, device="cuda", n_steps=None, collect_traces=0):
    """Returns accuracy at every eps in the sweep -- one model, a whole frontier."""
    model.eval()
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    # Default to the reasoner's OWN depth, not cfg.n_steps.  cfg.n_steps is TRAIL's
    # hop count; the recurrent baselines are built with n_steps * inner_steps
    # iterations so that every model gets the same belief-update budget.  Reading
    # cfg.n_steps here evaluated them at a third of the depth they were trained at
    # and silently flattered TRAIL.
    T = n_steps or int(getattr(model.reasoner, "n_steps", cfg.n_steps))
    # The sweep is over the OUTER threshold delta, which decides how many hops to
    # take.  The inner gap threshold eps decides how many mirror-descent
    # iterations each hop costs; it is held at cfg.eps and enters through the
    # update accounting, not through the answer.
    # Always evaluate the configured operating point, even when it is not one of the
    # swept values.  Previously a cfg.delta outside eps_sweep fell through to
    # `acc_full` and `float(T) * per_step`, which produces a plausible-looking row
    # that reports the architectural budget instead of what the model actually did.
    eps_list = sorted(set(list(cfg.eps_sweep) + [float(cfg.delta)]), reverse=True)
    correct = {e: 0 for e in eps_list}
    steps = {e: 0.0 for e in eps_list}
    updates = {e: 0.0 for e in eps_list}
    n, acc_full = 0, 0
    per_hop = {}
    halt_by_delta: dict = {}
    gap_curve = torch.zeros(T)
    inner_curve = None
    halt_vs_hops = []
    traces = []

    # Only TRAIL emits a certificate; for the baselines every eps must fall back
    # to their fixed budget, otherwise the sweep would "halt" them at step 1 on a
    # gap vector that is identically zero by construction.
    has_cert = bool(getattr(model.reasoner, "produces_certificate", False))
    # TRAIL counts hops, the baselines count iterations.  One hop costs
    # ``inner_steps`` belief updates, so the frontier is reported in *belief
    # updates* -- otherwise the x-axis of Fig. 4 would be comparing two units.
    per_step = int(getattr(model.reasoner, "inner_steps", 1))

    for batch in dl:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        st, V = model(batch, n_steps=T, record=True)
        logits = torch.stack(st.logits, dim=1)                 # (B, T+1, A)
        gaps = torch.stack(st.gap, dim=1) if (st.gap and has_cert) else None
        moves = torch.stack(st.move, dim=1) if (st.move and has_cert) else None
        inner = torch.stack(st.inner_gap, dim=1) if (st.inner_gap and has_cert) else None
        y = batch["answer"]
        B = y.shape[0]
        n += B
        acc_full += (logits[:, -1].argmax(-1) == y).sum().item()
        if gaps is not None:
            gap_curve[: gaps.shape[1]] += gaps.float().mean(0).cpu()[:T]
        if inner is not None:
            m = inner.float().mean(0).cpu()
            inner_curve = m if inner_curve is None else inner_curve + m

        # cost of each hop in mirror-descent iterations, under the inner rule
        if inner is not None:
            ig = inner.view(B, gaps.shape[1], -1)                      # (B,H,K)
            g0 = ig[:, :1, :1].clamp_min(1e-8)
            hit = ig <= (cfg.eps * g0)
            cost = torch.where(hit.any(-1), hit.float().argmax(-1) + 1.0,
                               torch.full_like(ig[..., 0], float(ig.shape[-1])))
        else:
            cost = None

        for e in eps_list:
            if moves is None or e <= 0:
                t_idx = torch.full((B,), logits.shape[1] - 1, device=device, dtype=torch.long)
            else:
                below = moves <= e
                first = torch.where(below.any(1), below.float().argmax(1),
                                    torch.full_like(y, moves.shape[1] - 1))
                t_idx = first.clamp(max=logits.shape[1] - 1)
            pick = logits[torch.arange(B, device=device), t_idx]
            correct[e] += (pick.argmax(-1) == y).sum().item()
            steps[e] += (t_idx + 1).float().sum().item()
            if cost is not None:
                keep = (torch.arange(cost.shape[1], device=device)[None, :] <= t_idx[:, None])
                updates[e] += (cost * keep).sum().item()
            else:
                updates[e] += (t_idx + 1).float().sum().item() * per_step
            if "hops" in batch and moves is not None:
                halt_by_delta.setdefault(e, []).append(
                    torch.stack([t_idx.float(), batch["hops"].float()], -1).cpu())

        if moves is not None and "hops" in batch:
            below = moves <= float(getattr(model.reasoner, "delta", 0.0))
            first = torch.where(below.any(1), below.float().argmax(1),
                                torch.full_like(y, moves.shape[1] - 1))
            halt_vs_hops.append(torch.stack([first.float(), batch["hops"].float()], -1).cpu())

        if "hops" in batch:
            ok = (logits[:, -1].argmax(-1) == y)
            for h in batch["hops"].unique():
                m = batch["hops"] == h
                k = int(h.item())
                c, t = per_hop.get(k, (0, 0))
                per_hop[k] = (c + ok[m].sum().item(), t + int(m.sum().item()))

        if collect_traces and len(traces) < collect_traces:
            P = torch.stack(st.p, dim=1).cpu()
            for b in range(min(B, collect_traces - len(traces))):
                traces.append({"p": P[b].numpy(),
                               "gap": gaps[b].cpu().numpy() if gaps is not None else None,
                               "answer": int(y[b].item()),
                               "hops": int(batch["hops"][b].item()) if "hops" in batch else -1})

    out: dict = {
        "n": n,
        "acc_full": acc_full / n,
        "acc_eps": correct[float(cfg.delta)] / n,
        "avg_steps": steps[float(cfg.delta)] / n,
        "updates_per_step": per_step,
        "avg_updates": updates[float(cfg.delta)] / n,
        "frontier": [{"delta": e, "acc": correct[e] / n, "steps": steps[e] / n,
                      "updates": updates[e] / n} for e in eps_list],
        "gap_curve": (gap_curve / max(1, len(dl))).tolist(),
        "inner_gap_curve": ((inner_curve / max(1, len(dl))).tolist()
                            if inner_curve is not None else None),
        "acc_by_hops": {k: c / t for k, (c, t) in sorted(per_hop.items()) if t > 0},
    }
    # Prop. 3 predicts halting time tracks difficulty.  Reporting it at one
    # threshold hides the failure mode: if the rule saturates, every example halts
    # at the same step and the correlation is undefined for lack of variance
    # rather than false.  So report rho *and* the spread at every threshold.
    out["halt_stats_by_delta"] = {}
    for e, chunks in halt_by_delta.items():
        if not chunks:
            continue
        hv = torch.cat(chunks).numpy()
        out["halt_stats_by_delta"][str(e)] = {
            "rho": _spearman(hv[:, 0], hv[:, 1]),
            "mean_halt": float(hv[:, 0].mean() + 1),
            "std_halt": float(hv[:, 0].std()),
        }
    if halt_vs_hops:
        hv = torch.cat(halt_vs_hops).numpy()
        out["halt_vs_hops"] = hv.tolist()[:20000]
        out["spearman_halt_hops"] = _spearman(hv[:, 0], hv[:, 1])
    if traces:
        out["traces"] = traces
    return out


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------
@torch.no_grad()
def measure_drift(model, ds, cfg, device="cuda", n_batches=4):
    """Fig. 3: distance of the visual state from conv(V), per reasoning step.

    Zero for TRAIL by Prop. 2; for free-form latent recursion it is whatever the
    network happens to do, which is the point.
    """
    model.eval()
    dl = DataLoader(ds, batch_size=min(cfg.batch_size, 64), shuffle=False)
    per_step = []
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        st, V = model(batch, record=True)
        ds_ = []
        for p in st.p:
            if p.shape[-1] == V.shape[1]:
                z = torch.einsum("bn,bnd->bd", p, V)
            else:
                continue
            ds_.append(cert.hull_drift(z, V).mean().item())
        if hasattr(st, "z_final"):
            ds_.append(cert.hull_drift(st.z_final.float(), V.float()).mean().item())
        per_step.append(ds_)
    L = min(len(x) for x in per_step)
    return np.mean([x[:L] for x in per_step], axis=0).tolist()
