"""Frozen-trunk feature cache, shared by CLEVR and GQA.

Featurising is the one expensive, one-off part of the pipeline, and it is what
makes a five-model comparison fit in a Kaggle session: run the trunk once, write
float16 atoms to a memmap, and let every reasoner train off the same cache.  That
also happens to be what makes the comparison fair -- nothing about the visual
front end can differ between rows of the table.

The only dataset-specific part is how an image id becomes a path, so that is a
callback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


@torch.no_grad()
def cache_features(
    cfg,
    image_ids: Iterable,
    path_of: Callable[[object], Path],
    tag: str,
    device: str = "cuda",
    batch_size: int = 32,
):
    """Returns (memmap of shape (n_images, N, C), {str(image_id): row}, C).

    Re-running with the same ``tag`` reuses the cache instead of recomputing it,
    so a crashed training run does not cost the featurisation again.
    """
    from PIL import Image
    from torchvision import transforms
    from tqdm.auto import tqdm

    from ..models.backbone import build_resnet_trunk

    os.makedirs(cfg.cache_dir, exist_ok=True)
    ids = sorted(image_ids, key=str)
    full_tag = f"{tag}_{cfg.features}_{cfg.image_size}_{len(ids)}"
    fpath = Path(cfg.cache_dir) / f"feat_{full_tag}.npy"
    ipath = Path(cfg.cache_dir) / f"idx_{full_tag}.json"

    trunk, ch = build_resnet_trunk(cfg.features, pretrained=True)
    N = cfg.grid * cfg.grid
    if fpath.exists() and ipath.exists():
        return np.load(fpath, mmap_mode="r"), json.load(open(ipath)), ch

    pos = {str(ix): i for i, ix in enumerate(ids)}
    trunk = trunk.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    mm = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.float16,
                                   shape=(len(ids), N, ch))
    buf, rows, missing = [], [], 0
    try:
        for ix in tqdm(ids, desc=f"featurising {tag}"):
            p = path_of(ix)
            if not Path(p).exists():
                missing += 1
                continue
            buf.append(tf(Image.open(p).convert("RGB")))
            rows.append(pos[str(ix)])
            if len(buf) == batch_size:
                out = trunk(torch.stack(buf).to(device))
                mm[rows] = out.flatten(2).transpose(1, 2).half().cpu().numpy()
                buf, rows = [], []
        if buf:
            out = trunk(torch.stack(buf).to(device))
            mm[rows] = out.flatten(2).transpose(1, 2).half().cpu().numpy()
        mm.flush()
    except BaseException:
        # a half-written cache is worse than none: the next run would train on zeros
        del mm
        fpath.unlink(missing_ok=True)
        raise

    if missing:
        print(f"warning: {missing}/{len(ids)} images not found under the resolved path; "
              f"their atoms are zero. Check Config.data_root.")
    json.dump(pos, open(ipath, "w"))
    return np.load(fpath, mmap_mode="r"), pos, ch
