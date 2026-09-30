"""Any-of pairs on the occlusion split: the v1.0 target sets stored with the evaluation set equal those recomputed from the shipped fruit tables (scripts/occ_anyof.py asserts this while it
runs), the plain reading only adds targets, and it coincides with v1.0 on the pairs that benchmark v1.1 would issue."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
oa = pytest.importorskip("occ_anyof")


def test_pairs_are_aligned_with_the_stored_targets():
    pairs = oa.any_of_pairs()                               # raises if a stored target set differs from the recomputed one
    assert len(pairs) == 66
    n_unamb = sum(p["unamb"] for p in pairs.values())
    assert 0 < n_unamb < len(pairs)
    for p in pairs.values():
        for old, new in zip(p["v10"], p["rev"]):
            assert set(old) <= set(new) and old
        if p["unamb"]:
            assert p["v10"] == p["rev"]


def test_score_follows_the_recorded_correctness():
    pairs = {0: dict(v10=[(1,), (2,)], rev=[(1, 3), (2,)], unamb=False, n=5), 1: dict(v10=[(0,), (4,)], rev=[(0,), (4,)], unamb=True, n=5)}

    def rec(i, which, sel, ok):
        return {"index": i, "which": which, "first_detached": sel, "target_correct": ok}
    recs = [rec(0, "plus", 3, False), rec(0, "minus", 2, True),          # the plain reading accepts fruit 3, the v1.0 targets do not
            rec(1, "plus", 0, True), rec(1, "minus", 4, True)]
    a, b, c, n, nu = oa.score(pairs, recs)
    assert (a, b, c, n, nu) == (50.0, 100.0, 100.0, 2, 1)
    with pytest.raises(AssertionError):                                  # a recorded correctness that does not follow from the stored targets is caught
        oa.score(pairs, [rec(0, "plus", 3, True), rec(0, "minus", 2, True)])
