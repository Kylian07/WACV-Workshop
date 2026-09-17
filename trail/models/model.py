"""End-to-end model: atoms -> question -> reasoner -> anytime answers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import AtomProjector, ScratchCNN, build_resnet_trunk
from .baselines import REASONERS
from .heads import AnswerHead
from .text import QuestionEncoder
from .trail_cell import TrailReasoner


class VQAModel(nn.Module):
    def __init__(self, cfg, vocab: int, n_answers: int, feat_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.trunk = None
        if cfg.features == "pixels":
            self.trunk = ScratchCNN(cfg.d_feat)
            feat_dim = cfg.d_feat
        assert feat_dim is not None, "feat_dim must be given when training on cached features"

        self.atoms = AtomProjector(feat_dim, cfg.d_vis, grid=cfg.grid, dropout=cfg.dropout)
        self.text = QuestionEncoder(vocab, cfg.d_ctrl, dropout=cfg.dropout)
        self.head = AnswerHead(cfg.d_vis, cfg.d_ctrl, n_answers, cfg.d_hidden,
                               use_count=cfg.use_count, dropout=cfg.dropout)

        if cfg.reasoner == "trail":
            self.reasoner = TrailReasoner(
                d_vis=cfg.d_vis, d_ctrl=cfg.d_ctrl, n_steps=cfg.n_steps,
                inner_steps=cfg.inner_steps, beta=cfg.beta, grid=cfg.grid,
                n_heads=cfg.n_heads, learn_eta=cfg.learn_eta,
                eta_safety=cfg.eta_safety, use_transport=cfg.use_transport,
                use_pos_bias=cfg.use_pos_bias, temp=cfg.temp,
                delta=cfg.delta, gap_mode=cfg.gap_mode, share_steps=cfg.share_steps,
            )
        else:
            # The iterative baselines get n_steps * inner_steps iterations, so every
            # model in the table is allowed the same number of belief updates.  FiLM
            # is not an iterative reasoner -- its "steps" are residual conv blocks --
            # so it keeps its standard depth of 4 and is reported as such.
            depth = 4 if cfg.reasoner == "film" else cfg.n_steps * cfg.inner_steps
            self.reasoner = REASONERS[cfg.reasoner](
                d_vis=cfg.d_vis, d_ctrl=cfg.d_ctrl, n_steps=depth, grid=cfg.grid,
                share_steps=cfg.share_steps,
            )

    def encode_visual(self, batch) -> torch.Tensor:
        if self.trunk is not None:
            with torch.set_grad_enabled(self.training):
                feat = self.trunk(batch["image"])
            return self.atoms(feat)
        return self.atoms(batch["feat"])

    def forward(self, batch, n_steps=None, eps=None, record=True):
        V = self.encode_visual(batch)
        words, q_vec, mask = self.text(batch["question"])
        return self.reasoner(V, q_vec, words, mask, n_steps=n_steps, eps=eps,
                             record=record, answer_head=self.head), V
