"""Visual atoms.

Two options, both giving a (B, N, d) bag of atoms on a grid:

  ``resnet``  frozen ImageNet ResNet-18/34 up to layer3 -> 14x14 x 256/1024.
              Features are cached to disk once and the reasoner trains on the
              cache, which is what makes the whole study fit in a Kaggle
              session.
  ``scratch`` the 4-layer CNN used by the from-scratch FiLM setting, trained
              end-to-end.  No pretrained weights, no download.

Every model in the comparison uses the same backbone and the same projection,
so parameter counts and FLOPs differ only in the reasoning module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScratchCNN(nn.Module):
    def __init__(self, d_out: int = 256):
        super().__init__()
        ch = [3, 64, 128, 128, d_out]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AtomProjector(nn.Module):
    """Feature map -> L2-normalised atoms + a learned 2-D position code.

    The atoms are unit-norm so that <V_i, W c> is a cosine-like score and the
    energy scale is comparable across images -- which matters because eps is a
    *single global threshold* shared by every example (Thm. A).
    """

    def __init__(self, d_in: int, d_out: int, grid: int = 14, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in + 2, d_out), nn.ELU(), nn.Dropout(dropout),
                                  nn.Linear(d_out, d_out))
        self.norm = nn.LayerNorm(d_out)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, grid), torch.linspace(-1, 1, grid), indexing="ij")
        self.register_buffer("pos", torch.stack([ys, xs], -1).reshape(grid * grid, 2), persistent=False)

    def forward(self, feat):                 # feat: (B, C, H, W) or (B, N, C)
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            feat = feat.flatten(2).transpose(1, 2)
        B, N, C = feat.shape
        if N <= self.pos.shape[0]:
            pos = self.pos[:N]
        else:                      # evaluated on a finer grid than it was built for
            g = int(round(N ** 0.5))
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, g, device=feat.device),
                                    torch.linspace(-1, 1, g, device=feat.device), indexing="ij")
            pos = torch.stack([ys, xs], -1).reshape(-1, 2)[:N]
        pos = pos.unsqueeze(0).expand(B, N, 2).to(feat.dtype)
        v = self.proj(torch.cat([feat, pos], -1))
        v = self.norm(v)
        return torch.nn.functional.normalize(v, dim=-1)


def build_resnet_trunk(name: str = "resnet18", pretrained: bool = True):
    """Frozen trunk up to layer3.  Returns (module, channels)."""
    import torchvision

    weights = "IMAGENET1K_V1" if pretrained else None
    net = getattr(torchvision.models, name)(weights=weights)
    trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
    ch = {"resnet18": 256, "resnet34": 256, "resnet50": 1024, "resnet101": 1024}[name]
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk.eval(), ch
