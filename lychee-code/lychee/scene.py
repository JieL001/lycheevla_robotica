"""Scene-level sampling of the *symbolic* per-fruit view used for benchmark design statistics."""
from __future__ import annotations

import numpy as np

from .lang import FruitView


def sample_fruits(rng: np.random.Generator, n: int, *, mat_probs=None, u_range=(0.08, 0.92),
                  depth_range=(0.5, 0.9), vis_range=(0.2, 1.0)) -> list[FruitView]:
    """n fruits with maturity ~ scene-level Dirichlet mix, uniform image x / depth, Beta-ish visibility."""
    probs = rng.dirichlet(np.ones(3)) if mat_probs is None else np.asarray(mat_probs, float)
    mats = rng.choice(3, size=n, p=probs)
    us = rng.uniform(*u_range, size=n)
    ds = rng.uniform(*depth_range, size=n)
    vs = rng.uniform(*vis_range, size=n)
    return [FruitView(i, int(mats[i]), float(us[i]), float(ds[i]), float(vs[i])) for i in range(n)]
