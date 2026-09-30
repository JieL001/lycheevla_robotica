"""Marginal-matched control set (scripts/make_matched_dataset.py): the labels of a scene can be recomputed from the stored fruit table, and the raking matches the template frequencies."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from lychee import lang
from lychee.modular import parse_command
from lychee.paths import eval_meta

make_matched = pytest.importorskip("make_matched_dataset")


def test_stored_fruit_table_reproduces_the_labels():
    """The valid (template, target set) pairs recomputed from the stored image-space fruit table contain every stored command with its stored target set (IID evaluation scenes)."""
    d = np.load(eval_meta("iid_600"))
    idx = {s: k for k, s in enumerate(lang.all_specs())}
    n = 0
    for i in np.flatnonzero(d["ok"])[:40]:
        valid = dict(make_matched._valid((d["fxy"][i], d["frad"][i], d["fmat"][i], d["fvis"][i], d["fvalid"][i])))
        for k in range(int(d["ncmd"][i])):
            spec = parse_command(d["text"][i, k])
            assert idx[spec] in valid, (i, k, d["text"][i, k])
            assert tuple(sorted(valid[idx[spec]])) == tuple(int(j) for j in np.flatnonzero(d["tmask"][i, k]))
            n += 1
    assert n >= 60


def test_raking_matches_the_target_frequencies():
    rng = np.random.RandomState(0)
    S, N = 12, 3000
    V = rng.rand(N, S) < 0.4
    V[np.arange(N), rng.choice([s for s in range(S) if s != 3], N)] = True      # every scene has at least one valid template that the target uses (3 is never used)
    natural = (V * 1.0 / V.sum(1, keepdims=True)).sum(0)                # expected counts of a uniform draw
    target = natural[::-1] * (N / natural.sum())                       # another mix (the natural one reversed)
    target[3] = 0.0                                                     # a template that the target never uses
    target *= N / target.sum()
    w, err, n_it = make_matched.rake(V, target, iters=500, tol=0.05)
    z = (V * w).sum(1, keepdims=True)
    e = (V * w / z).sum(0)
    assert err < 0.05 and np.abs(e - target).max() < 0.05
    assert w[3] == 0.0 and e[3] == 0.0
    # a draw from p(template | scene) proportional to w gives counts within multinomial noise of the target
    draws = np.array([rng.choice(S, p=(V[r] * w) / (V[r] * w).sum()) for r in range(N)])
    counts = np.bincount(draws, minlength=S)
    assert np.all(np.abs(counts - target) < 5 * np.sqrt(np.maximum(target, 1)) + 1)
