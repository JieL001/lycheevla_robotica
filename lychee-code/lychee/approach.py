"""Geometric clearance planning for the scripted expert (numpy only).

The Panda hand is approximated by three oriented boxes in the tool frame (z = approach axis toward the
object, y = finger-opening axis, TCP at the origin): two finger pads and the 20 cm wide palm.  A
candidate approach (direction + wrist roll) is scored by the smallest gap between any box and any
*non-target* fruit sphere along the straight TCP path from the pre-grasp pose to the grasp pose.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

FRUIT_R = 0.030
OPEN_HALF, CLOSED_HALF = 0.045, 0.033           # finger-pad centre offset while opening / closed on a fruit
PALM = (np.array([0.0, 0.0, -0.0744]), np.array([0.03, 0.10, 0.029]))


def _fingers(half):
    h = np.array([0.011, 0.011, 0.0227])
    return [(np.array([0.0, +half, -0.0225]), h), (np.array([0.0, -half, -0.0225]), h)]


def tool_rot(a: np.ndarray, roll_deg: float) -> Rot:
    a = a / np.linalg.norm(a)
    f0 = np.cross([0, 0, 1.0], a)
    f0 = f0 / np.linalg.norm(f0)
    f = Rot.from_rotvec(a * np.deg2rad(roll_deg)).apply(f0)
    x = np.cross(f, a)
    return Rot.from_matrix(np.stack([x, f, a], axis=1))


def gap(R: Rot, tcp_path: np.ndarray, others: np.ndarray, half: float) -> float:
    """Smallest surface-to-box gap (m) over the path; negative = penetration."""
    if len(others) == 0:
        return 1.0
    Rm = R.as_matrix()
    best = 1e9
    boxes = _fingers(half) + [PALM]
    for tcp in tcp_path:
        local = (others - tcp) @ Rm                       # (M,3): coordinates in the tool frame
        for c, h in boxes:
            d = np.maximum(np.abs(local - c) - h, 0.0)
            best = min(best, float(np.linalg.norm(d, axis=1).min()) - FRUIT_R)
    return best


APPROACH_DIRS = [np.array(v, float) for v in [(1, 0, 0)]] + [
    Rot.from_euler(ax, s * 20, degrees=True).apply([1.0, 0, 0]) for ax in "zy" for s in (+1, -1)]
ROLLS = (0.0, 90.0, 45.0, -45.0)


G_SAFE = 0.015                                   # clearance (m) considered "enough"; then prefer the cheapest motion


def _pick(cands):
    """cands: list of (cost, gap, payload). First candidate (by cost) with gap >= G_SAFE, else the widest gap."""
    ok = [c for c in sorted(cands, key=lambda c: c[0]) if c[1] >= G_SAFE]
    return ok[0] if ok else max(cands, key=lambda c: c[1])


def plan_approach(rest, t: int, pre_dist: float = 0.14):
    """(approach dir, roll, tool rotation, gap) for target ``t``."""
    tgt = np.asarray(rest[t]); others = np.array([r for i, r in enumerate(rest) if i != t]).reshape(-1, 3)
    cands = []
    for k, a in enumerate(APPROACH_DIRS):
        for roll in ROLLS:
            R = tool_rot(a, roll)
            path = np.array([tgt - (pre_dist - s) * a for s in np.arange(0.0, pre_dist, 0.01)])
            cost = (k > 0) + {0.0: 0, 45.0: 1, -45.0: 1, 90.0: 2}[roll]
            cands.append((cost, gap(R, path, others, OPEN_HALF), (a, roll, R)))
    cost, g, (a, roll, R) = _pick(cands)
    return a, roll, R, g


PULL_DEFAULT = np.array([-0.35, 0.0, -0.94]) / np.linalg.norm([-0.35, 0.0, -0.94])
PULL_DIRS = [PULL_DEFAULT] + [Rot.from_euler(ax, s * 35, degrees=True).apply(PULL_DEFAULT) for ax in "zy" for s in (+1, -1)]


def plan_pull(rest, t: int, R: Rot, dist: float = 0.07):
    tgt = np.asarray(rest[t]); others = np.array([r for i, r in enumerate(rest) if i != t]).reshape(-1, 3)
    cands = []
    for k, d in enumerate(PULL_DIRS):
        path = np.array([tgt + s * d for s in np.arange(0.0, dist, 0.01)])
        cands.append((k, gap(R, path, others, CLOSED_HALF), d))
    cost, g, d = _pick(cands)
    return d, g
