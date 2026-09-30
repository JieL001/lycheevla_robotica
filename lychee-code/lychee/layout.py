"""Scene layout for LycheeHarvest (numpy only, no simulator).

A ``Layout`` is everything the simulator needs to build one scene: fruit positions/maturities, leaf
poses, the camera and small robot-state noise.  It is generated from a single integer ``scene_seed``,
so the *same* scene can be replayed with different instructions (counterfactual pairs).

Visibility is defined analytically: the projected-area-weighted fraction of a fruit's camera-facing
surface that is not hidden by a leaf or another fruit (ray casting from the camera).  It is the label
for the visibility head and for the occlusion splits; ``scripts/check_visibility.py`` compares it with
the rendered segmentation.  Occluders are placed on purpose: for each fruit chosen to be occluded a
leaf is put on the camera->fruit ray and shifted sideways until the fruit has a requested visibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .lang import FruitView

R_FRUIT = 0.030
BRANCH = np.array([-0.05, 0.0, 0.46])
LEAF_HALF = np.array([0.05, 0.022])          # in-plane half sizes (length, width) of a leaf
Y_RANGE, Z_RANGE, X_JITTER = (-0.22, 0.22), (0.28, 0.40), 0.05
MIN_SPACING = 2.4 * R_FRUIT


@dataclass(frozen=True)
class Camera:
    eye: np.ndarray
    target: np.ndarray
    fov: float                                 # full field of view (rad), square image

    def basis(self):
        f = self.target - self.eye
        f = f / np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1.0])
        right = right / np.linalg.norm(right)
        return f, right, np.cross(right, f)

    def project(self, p: np.ndarray):
        """(u, depth, v): u = 0 at the image's left edge, v = 0 at the top."""
        f, right, up = self.basis()
        v = np.asarray(p) - self.eye
        z = float(v @ f)
        t = np.tan(self.fov / 2)
        return 0.5 + 0.5 * ((v @ right) / z) / t, z, 0.5 - 0.5 * ((v @ up) / z) / t


DEFAULT_CAM = Camera(np.array([-0.50, 0.0, 0.85]), np.array([-0.05, 0.0, 0.34]), np.deg2rad(55.0))


# ------------------------------------------------------------------ visual domain randomisation
@dataclass(frozen=True)
class DR:
    """Visual randomisation of one scene: camera pose/fov jitter, ambient light, colour tint and colour gains."""
    cam_dpos: np.ndarray
    cam_dtarget: np.ndarray
    fov_delta: float
    ambient: float
    tint: np.ndarray
    fruit_gain: float
    leaf_gain: float


DR_OFF = DR(np.zeros(3), np.zeros(3), 0.0, 0.30, np.ones(3), 1.0, 1.0)


def _two_sided(rng, lo, hi, size=None):
    """Uniform on [-hi, -lo] U [lo, hi] (used to make the held-out ranges disjoint from the training ranges)."""
    return rng.uniform(lo, hi, size=size) * rng.choice([-1, 1], size=size)


def sample_dr(seed: int, level: str) -> DR:
    """level: 'off' | 'train' | 'heldout'.  The held-out ranges do not overlap the training ranges."""
    if level == "off":
        return DR_OFF
    rng = np.random.RandomState((seed * 7 + 3) % (2 ** 31 - 1))
    if level == "train":
        return DR(np.clip(rng.normal(0, 0.012, 3), -0.025, 0.025), np.clip(rng.normal(0, 0.01, 3), -0.02, 0.02),
                  float(np.deg2rad(rng.uniform(-2, 2))), float(rng.uniform(0.25, 0.40)), 1 + rng.uniform(-0.06, 0.06, 3),
                  float(rng.uniform(0.9, 1.1)), float(rng.uniform(0.9, 1.1)))
    if level != "heldout":
        raise ValueError(level)
    pick = lambda a, b: float(rng.uniform(*a) if rng.rand() < 0.5 else rng.uniform(*b))
    return DR(_two_sided(rng, 0.04, 0.08, 3), _two_sided(rng, 0.03, 0.06, 3), float(np.deg2rad(_two_sided(rng, 3.5, 6.0))),
              pick((0.10, 0.20), (0.55, 0.80)), 1 + _two_sided(rng, 0.12, 0.25, 3),
              pick((0.65, 0.80), (1.20, 1.40)), pick((0.65, 0.80), (1.20, 1.40)))


def camera_for(dr: DR, base: Camera = DEFAULT_CAM) -> Camera:
    return Camera(base.eye + dr.cam_dpos, base.target + dr.cam_dtarget, base.fov + dr.fov_delta)


@dataclass
class Leaf:
    pos: np.ndarray
    quat_wxyz: np.ndarray
    shade: float = 0.5
    occluder: bool = False                     # placed deliberately in front of a fruit

    def axes(self):
        q = self.quat_wxyz
        R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return R[:, 0], R[:, 1], R[:, 2]        # length axis, width axis, normal


@dataclass
class Layout:
    seed: int
    mats: np.ndarray                            # (n,) maturity ids
    pos: np.ndarray                             # (n,3) fruit centres
    leaves: list                                # list[Leaf]
    cam: Camera = DEFAULT_CAM
    q_noise: np.ndarray = field(default_factory=lambda: np.zeros(7))
    vis: Optional[np.ndarray] = None            # (n,) analytic visibility
    dr: DR = DR_OFF

    @property
    def n(self) -> int:
        return len(self.mats)

    def fruit_views(self) -> list:
        out = []
        for i in range(self.n):
            u, d, _ = self.cam.project(self.pos[i])
            out.append(FruitView(i, int(self.mats[i]), float(u), float(d), float(self.vis[i])))
        return out


# ------------------------------------------------------------------ visibility (ray casting)
def _fibonacci(m: int) -> np.ndarray:
    k = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * k / m)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


_SPHERE = _fibonacci(384)


def fruit_visibility(pos: np.ndarray, leaves: Sequence[Leaf], cam: Camera, only: Optional[Sequence[int]] = None) -> np.ndarray:
    """Projected-area-weighted visible fraction of each fruit (leaves and other fruits both occlude)."""
    n = len(pos)
    idx = range(n) if only is None else only
    out = np.full(n, np.nan)
    leaf_data = [(lf.pos, *lf.axes()) for lf in leaves]
    for i in idx:
        to_cam = cam.eye - pos[i]
        to_cam = to_cam / np.linalg.norm(to_cam)
        w = _SPHERE @ to_cam
        keep = w > 1e-3
        nrm, w = _SPHERE[keep], w[keep]
        q = pos[i] + R_FRUIT * 1.002 * nrm            # nudge outward so the point never hits its own sphere
        d = q - cam.eye
        dist = np.linalg.norm(d, axis=1)
        d = d / dist[:, None]
        blocked = np.zeros(len(q), bool)
        for c, u, v, nl in leaf_data:                 # leaf = thin rectangle
            denom = d @ nl
            ok = np.abs(denom) > 1e-6
            t = np.where(ok, ((c - cam.eye) @ nl) / np.where(ok, denom, 1.0), -1.0)
            hit = ok & (t > 1e-4) & (t < dist - 1e-4)
            x = cam.eye + t[:, None] * d - c
            blocked |= hit & (np.abs(x @ u) <= LEAF_HALF[0]) & (np.abs(x @ v) <= LEAF_HALF[1])
        for j in range(n):                             # other fruits
            if j == i:
                continue
            oc = cam.eye - pos[j]
            b = d @ oc
            disc = b * b - (oc @ oc - R_FRUIT ** 2)
            t = -b - np.sqrt(np.maximum(disc, 0.0))
            blocked |= (disc > 0) & (t > 1e-4) & (t < dist - 1e-4)
        out[i] = 1.0 - float((w * blocked).sum() / w.sum())
    return out


# ------------------------------------------------------------------ sampling
def _sample_fruits(rng: np.random.RandomState, n: int):
    P = []
    for _ in range(4000):
        if len(P) == n:
            break
        p = np.array([BRANCH[0] + rng.uniform(-X_JITTER, X_JITTER), rng.uniform(*Y_RANGE), rng.uniform(*Z_RANGE)])
        if all(np.linalg.norm(p - q) > MIN_SPACING for q in P):
            P.append(p)
    return np.array(P)


def _quat_from_axes(u, v, nrm):
    R = np.stack([u, v, nrm], axis=1)
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _place_occluder(rng, pos, i, leaves, cam, v_target, tol=0.06):
    """Add one leaf in front of fruit ``i`` and shift it sideways until the fruit's visibility ~ v_target."""
    p = pos[i]
    to_cam = cam.eye - p
    to_cam = to_cam / np.linalg.norm(to_cam)
    s = rng.uniform(0.03, 0.11)
    c0 = p + s * to_cam
    nrm = cam.eye - c0
    nrm = nrm / np.linalg.norm(nrm)
    tilt = Rot.from_rotvec(rng.normal(size=3) * 0.35).apply(nrm)          # ~20 degrees of random tilt
    tilt = tilt / np.linalg.norm(tilt)
    a = np.cross(tilt, rng.normal(size=3))
    a = a / np.linalg.norm(a)
    b = np.cross(tilt, a)
    ang = rng.uniform(0, 2 * np.pi)
    u, v = np.cos(ang) * a + np.sin(ang) * b, -np.sin(ang) * a + np.cos(ang) * b
    f, right, up = cam.basis()
    phi = rng.uniform(0, 2 * np.pi)
    side = np.cos(phi) * right + np.sin(phi) * up                        # direction of the lateral shift
    q = _quat_from_axes(u, v, tilt)
    lo, hi = 0.0, 0.14
    best = None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lf = Leaf(c0 + mid * side, q, rng.uniform(0.35, 0.65), occluder=True)
        vis = fruit_visibility(pos, leaves + [lf], cam, only=[i])[i]
        best = (lf, vis)
        if abs(vis - v_target) <= tol:
            break
        if vis < v_target:                      # too hidden -> move the leaf away
            lo = mid
        else:
            hi = mid
    return best


def sample_layout(seed: int, n: Optional[int] = None, n_range=(3, 10), leaf_range=(8, 30), p_occlude: float = 0.35,
                  occ_vis_range=(0.2, 1.0), max_leaves: int = 48, dr: str = "off") -> Layout:
    """Deterministic layout for ``seed``.  ``p_occlude`` = share of fruits that get a deliberate occluder whose
    target visibility is drawn from ``occ_vis_range``; ``leaf_range`` = number of additional 'decoration' leaves."""
    rng = np.random.RandomState(seed)
    dr_params = sample_dr(seed, dr)
    cam = camera_for(dr_params)
    n = int(rng.randint(n_range[0], n_range[1] + 1)) if n is None else n
    probs = rng.dirichlet(np.ones(3))
    mats = rng.choice(3, size=n, p=probs)
    pos = _sample_fruits(rng, n)
    n = len(pos)
    mats = mats[:n]
    leaves: list = []
    for i in range(n):
        if rng.rand() < p_occlude:
            vt = rng.uniform(*occ_vis_range)
            lf, _ = _place_occluder(rng, pos, i, leaves, cam, vt)
            leaves.append(lf)
    n_bg = int(rng.randint(leaf_range[0], leaf_range[1] + 1))
    before = fruit_visibility(pos, leaves, cam)
    for _ in range(n_bg):
        if len(leaves) >= max_leaves:
            break
        front = rng.rand() < 0.6
        c = np.array([BRANCH[0] + (rng.uniform(-0.16, -0.02) if front else rng.uniform(0.10, 0.22)),
                      rng.uniform(-0.30, 0.30), rng.uniform(0.22, 0.52)])
        q = Rot.from_euler("zyx", [rng.uniform(-np.pi, np.pi), rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)]).as_quat()
        lf = Leaf(c, np.array([q[3], q[0], q[1], q[2]]), rng.uniform(0.35, 0.65))
        # decoration leaves may only slightly reduce a fruit's visibility (keeps occlusion under control)
        after = fruit_visibility(pos, leaves + [lf], cam)
        if np.all(before - after <= 0.06):
            leaves.append(lf)
            before = after
    lay = Layout(seed, mats, pos, leaves, cam, rng.normal(0, 0.02, 7), dr=dr_params)
    lay.vis = fruit_visibility(pos, leaves, cam)
    return lay
