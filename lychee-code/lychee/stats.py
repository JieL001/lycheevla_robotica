"""Paired comparisons between two arms evaluated on the same scenes (numpy only).

The unit is the scene.  ``scene_outcomes`` maps evaluation records to one number per scene; ``compare`` returns the paired
difference with a percentile-bootstrap interval, McNemar's exact test for binary outcomes and the discordant counts;
``holm`` adjusts a family of p-values.  ``cluster_bootstrap`` resamples seeds and then scenes within seeds, for arms trained with
several seeds (the interval then describes the training recipe and not only the trained models).
"""
from __future__ import annotations

from math import comb
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from . import evalkit


def scene_outcomes(records: list, key: str = "PTA_sel") -> Dict[tuple, float]:
    """{(split, index): outcome in [0,1]} for a pair-level key ('PTA_sel', 'PTA_grasp', 'PTA_succ') or a per-command key
    ('TSA', 'success', 'wrong_target', 'clean_success': the mean over the two commands of the scene)."""
    pt = evalkit.pair_table(records)
    grasp = evalkit._grasp_ok
    pair_fns = {"PTA_sel": lambda v: v["plus"]["target_correct"] and v["minus"]["target_correct"],
                "PTA_succ": lambda v: v["plus"]["success"] and v["minus"]["success"],
                "PTA_grasp": lambda v: grasp(v["plus"]) and grasp(v["minus"]),
                "collapse": lambda v: v["plus"]["first_detached"] == v["minus"]["first_detached"] and v["plus"]["first_detached"] >= 0}
    cmd_fns = {"TSA": lambda r: r["target_correct"], "success": lambda r: r["success"], "wrong_target": lambda r: r["wrong_target"],
               "clean_success": lambda r: r["success"] and not r["touched_nontarget"]}
    if key in pair_fns:
        return {k: float(pair_fns[key](v)) for k, v in pt.items()}
    fn = cmd_fns[key]
    return {k: 0.5 * (float(fn(v["plus"])) + float(fn(v["minus"]))) for k, v in pt.items()}


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b01: only B correct, b10: only A correct)."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, p))


def compare(a: Dict[tuple, float], b: Dict[tuple, float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired difference A - B over the common scenes."""
    keys = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    d = x - y
    rng = np.random.RandomState(seed)
    boots = d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(axis=1) if len(d) else np.array([np.nan])
    binary = set(np.unique(np.concatenate([x, y]))) <= {0.0, 1.0}
    b10, b01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return dict(n=len(keys), mean_a=float(x.mean()) if len(x) else float("nan"), mean_b=float(y.mean()) if len(y) else float("nan"),
                diff=float(d.mean()) if len(d) else float("nan"), lo=float(np.quantile(boots, alpha / 2)), hi=float(np.quantile(boots, 1 - alpha / 2)),
                p_mcnemar=mcnemar_exact(b01, b10) if binary else None, b10=b10, b01=b01)


def holm(pvals: Sequence[float]) -> list:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bh(pvals: Sequence[float]) -> list:
    """Benjamini-Hochberg adjusted p-values (false-discovery control for the exploratory comparisons)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    prev = 1.0
    m = len(p)
    for rank, i in enumerate(order):
        prev = min(prev, p[i] * m / (m - rank))
        adj[i] = prev
    return adj.tolist()


def cluster_bootstrap(per_seed: Sequence[Dict[tuple, float]], n_boot: int = 4000, seed: int = 0, alpha: float = 0.05):
    """Mean over seeds with an interval that resamples seeds and, within each drawn seed, scenes.  ``per_seed`` = one scene->outcome
    dict per training seed (same scenes)."""
    keys = sorted(set.intersection(*[set(d) for d in per_seed]))
    M = np.array([[d[k] for k in keys] for d in per_seed])              # (seeds, scenes)
    rng = np.random.RandomState(seed)
    S, N = M.shape
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.randint(0, S, S)
        boots[i] = np.mean([M[j, rng.randint(0, N, N)].mean() for j in s])
    return float(M.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)), M.mean(axis=1).tolist()


def cluster_diff(per_seed_a: Sequence[Dict[tuple, float]], per_seed_b: Sequence[Dict[tuple, float]], n_boot: int = 4000, seed: int = 0,
                 alpha: float = 0.05):
    """Recipe-level difference A - B for arms trained with several seeds: seeds of A and of B are resampled independently, scenes jointly
    (both arms are evaluated on the same scenes).  Returns (mean difference, lo, hi, per-seed means of A, per-seed means of B)."""
    keys = sorted(set.intersection(*[set(d) for d in list(per_seed_a) + list(per_seed_b)]))
    A = np.array([[d[k] for k in keys] for d in per_seed_a])
    B = np.array([[d[k] for k in keys] for d in per_seed_b])
    rng = np.random.RandomState(seed)
    Sa, Sb, N = len(A), len(B), A.shape[1]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, N, N)
        ia, ib = rng.randint(0, Sa, Sa), rng.randint(0, Sb, Sb)
        boots[i] = A[ia][:, idx].mean() - B[ib][:, idx].mean()
    return (float(A.mean() - B.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)),
            A.mean(axis=1).tolist(), B.mean(axis=1).tolist())


def cluster_diff_indep(per_seed_a: Sequence[Dict[tuple, float]], per_seed_b: Sequence[Dict[tuple, float]], n_boot: int = 4000, seed: int = 0,
                       alpha: float = 0.05):
    """Difference A - B where A and B were measured on DIFFERENT scene sets (for example one arm on the saliency-reversed and on the IID scenes): the seeds and the
    scenes of A and of B are all resampled independently.  Returns (difference, lo, hi)."""
    ka = sorted(set.intersection(*[set(d) for d in per_seed_a]))
    kb = sorted(set.intersection(*[set(d) for d in per_seed_b]))
    A = np.array([[d[k] for k in ka] for d in per_seed_a])
    B = np.array([[d[k] for k in kb] for d in per_seed_b])
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ia, ib = rng.randint(0, len(A), len(A)), rng.randint(0, len(B), len(B))
        boots[i] = A[ia][:, rng.randint(0, A.shape[1], A.shape[1])].mean() - B[ib][:, rng.randint(0, B.shape[1], B.shape[1])].mean()
    return float(A.mean() - B.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def cluster_did(a_sal: Sequence[Dict[tuple, float]], a_iid: Sequence[Dict[tuple, float]], b_sal: Sequence[Dict[tuple, float]], b_iid: Sequence[Dict[tuple, float]],
                n_boot: int = 4000, seed: int = 0, alpha: float = 0.05):
    """Difference in differences (A_sal - A_iid) - (B_sal - B_iid) of two arms on two scene sets ('sal' and 'iid'): the scenes of each set are resampled once and shared
    by the two arms, the seeds of each arm once and shared by its two sets (the same networks were evaluated on both sets).  Returns (did, lo, hi)."""
    ks = sorted(set.intersection(*[set(d) for d in list(a_sal) + list(b_sal)]))
    ki = sorted(set.intersection(*[set(d) for d in list(a_iid) + list(b_iid)]))
    AS, AI = np.array([[d[k] for k in ks] for d in a_sal]), np.array([[d[k] for k in ki] for d in a_iid])
    BS, BI = np.array([[d[k] for k in ks] for d in b_sal]), np.array([[d[k] for k in ki] for d in b_iid])
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s_idx, i_idx = rng.randint(0, len(ks), len(ks)), rng.randint(0, len(ki), len(ki))
        ia, ib = rng.randint(0, len(AS), len(AS)), rng.randint(0, len(BS), len(BS))
        boots[i] = (AS[ia][:, s_idx].mean() - AI[ia][:, i_idx].mean()) - (BS[ib][:, s_idx].mean() - BI[ib][:, i_idx].mean())
    est = (AS.mean() - AI.mean()) - (BS.mean() - BI.mean())
    return float(est), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
