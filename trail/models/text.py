"""Question encoder shared by every model in the comparison."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    def __init__(self, vocab: int, d_ctrl: int = 512, n_layers: int = 1, dropout: float = 0.15):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_ctrl, padding_idx=0)
        self.rnn = nn.LSTM(d_ctrl, d_ctrl // 2, n_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_ctrl)

    def forward(self, tokens: torch.Tensor):
        mask = tokens.ne(0)
        lengths = mask.sum(-1).clamp_min(1).cpu()
        x = self.drop(self.emb(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h, _) = self.rnn(packed)
        words, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=tokens.shape[1])
        q = torch.cat([h[-2], h[-1]], dim=-1)
        return self.norm(words), self.norm(q), mask
