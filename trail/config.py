"""Single source of truth for every experiment in the paper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # ---- data -----------------------------------------------------------
    dataset: str = "clevr"            # clevr | gqa | synthetic
    data_root: str | None = None      # auto-discovered under /kaggle/input when None
    cache_dir: str = "/kaggle/working/trail_cache"
    features: str = "resnet18"        # resnet18 | resnet34 | pixels
    image_size: int = 224
    grid: int = 14                    # sqrt(N); N = grid**2 atoms
    max_train_images: int = 0         # 0 = all
    max_train_questions: int = 0
    max_val_questions: int = 0
    max_q_len: int = 46
    split: str = "iid"                # iid | depth | hoc
    depth_train_max: int = 10         # 'depth' split: train on programs <= this
    depth_test_min: int = 15          # ... test on programs >= this
    hoc_heldout: tuple = ("red", "cube")

    # ---- model ----------------------------------------------------------
    reasoner: str = "trail"           # trail | attn1 | latent_freeform | mac | film
    d_feat: int = 256
    d_vis: int = 256
    d_ctrl: int = 256
    d_hidden: int = 512
    n_steps: int = 4                  # outer loop: reasoning hops
    inner_steps: int = 4              # inner loop: mirror-descent iterations per hop
    n_heads: int = 4
    beta: float = 1.0                 # relational weight; eta = 1/(4 beta)
    temp: float = 1.0
    learn_eta: bool = False           # ablation: ignore the prescribed step size
    eta_safety: float = 1.0
    use_transport: bool = True        # ablation: beta = 0 (pure MWU over atoms)
    use_pos_bias: bool = True
    use_count: bool = True
    dropout: float = 0.15

    # ---- halting --------------------------------------------------------
    gap_mode: str = "relative"        # certificate on G_t / G_0 (scale-free) or raw G_t
    delta: float = 1e-4               # outer rule: stop when a hop moves belief by < delta
    eps: float = 0.10                 # certificate threshold used at test time
    eps_sweep: tuple = (0.9, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05, 0.0)
    anytime_loss: str = "uniform"     # uniform | last | linear

    # ---- optimisation ---------------------------------------------------
    epochs: int = 12
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 500
    grad_clip: float = 5.0
    amp: bool = True
    num_workers: int = 2
    seed: int = 0

    # ---- bookkeeping ----------------------------------------------------
    out_dir: str = "/kaggle/working/runs/trail"
    smoke: bool = False
    log_every: int = 100

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def smoke_test(cls, **kw):
        """Tiny end-to-end run: finishes in a couple of minutes on a Kaggle CPU."""
        base = dict(
            smoke=True, max_train_images=200, max_train_questions=2000,
            max_val_questions=1000, epochs=1, batch_size=32, n_steps=3, inner_steps=2,
            grid=7, image_size=112, d_vis=128, d_ctrl=128, d_hidden=256,
        )
        base.update(kw)
        return cls(**base)
