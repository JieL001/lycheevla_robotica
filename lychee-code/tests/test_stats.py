import numpy as np
import pytest

from lychee import stats


def test_mcnemar_exact_known_values():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(5, 5) == 1.0
    assert abs(stats.mcnemar_exact(2, 10) - 2 * (1 + 12 + 66) / 4096) < 1e-12
    assert stats.mcnemar_exact(0, 20) == pytest.approx(2 / 2 ** 20)


def test_holm_and_bh_known_values():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert stats.holm([0.5]) == [0.5]


def _dict(vals):
    return {("iid", i): float(v) for i, v in enumerate(vals)}


def test_compare_perfect_separation_and_identity():
    a, b = _dict([1] * 20), _dict([0] * 20)
    r = stats.compare(a, b, n_boot=500)
    assert r["diff"] == 1.0 and r["lo"] == 1.0 and r["hi"] == 1.0 and r["b10"] == 20 and r["p_mcnemar"] < 1e-5
    r = stats.compare(a, a, n_boot=500)
    assert r["diff"] == 0.0 and r["p_mcnemar"] == 1.0


def test_compare_uses_only_common_scenes_and_skips_mcnemar_for_non_binary():
    a, b = _dict([1, 0, 1, 1]), {("iid", 1): 0.0, ("iid", 2): 0.0, ("iid", 3): 1.0, ("iid", 9): 1.0}
    r = stats.compare(a, b, n_boot=200)
    assert r["n"] == 3 and abs(r["diff"] - 1 / 3) < 1e-9          # common scenes 1, 2, 3: a = (0, 1, 1), b = (0, 0, 1)
    assert stats.compare(_dict([0.5, 1.0]), _dict([0.0, 1.0]), n_boot=100)["p_mcnemar"] is None


def test_scene_outcomes_from_records():
    def rec(i, which, targets, first, touched=False):
        ok = first in targets
        return dict(split="iid", index=i, which=which, targets=targets, first_detached=first, first_grasped=first, target_correct=ok,
                    success=ok, wrong_target=first >= 0 and not ok, touched_nontarget=touched, n_fruit=5, family="mat_side")
    recs = [rec(0, "plus", [0], 0), rec(0, "minus", [1], 1), rec(1, "plus", [0], 0), rec(1, "minus", [1], 0, touched=True)]
    assert stats.scene_outcomes(recs, "PTA_sel") == {("iid", 0): 1.0, ("iid", 1): 0.0}
    assert stats.scene_outcomes(recs, "TSA") == {("iid", 0): 1.0, ("iid", 1): 0.5}
    assert stats.scene_outcomes(recs, "collapse") == {("iid", 0): 0.0, ("iid", 1): 1.0}
    assert stats.scene_outcomes(recs, "clean_success") == {("iid", 0): 1.0, ("iid", 1): 0.5}


def test_cluster_bootstrap_interval_is_wider_when_seeds_disagree():
    rng = np.random.RandomState(0)
    base = rng.rand(80) < 0.5
    same = [_dict(base.astype(float))] * 3
    m1, lo1, hi1, _ = stats.cluster_bootstrap(same, n_boot=400)
    diff = [_dict(base.astype(float)), _dict((rng.rand(80) < 0.3).astype(float)), _dict((rng.rand(80) < 0.7).astype(float))]
    m2, lo2, hi2, per = stats.cluster_bootstrap(diff, n_boot=400)
    assert (hi2 - lo2) > (hi1 - lo1) and len(per) == 3


def test_cluster_diff_recovers_a_known_gap_and_widens_with_seed_noise():
    rng = np.random.RandomState(1)
    a = [_dict((rng.rand(200) < 0.9).astype(float)) for _ in range(3)]
    b = [_dict((rng.rand(200) < 0.7).astype(float)) for _ in range(3)]
    d, lo, hi, ma, mb = stats.cluster_diff(a, b, n_boot=600)
    assert 0.1 < d < 0.3 and lo < d < hi and lo > 0 and len(ma) == len(mb) == 3
    d2, lo2, hi2, *_ = stats.cluster_diff(a, a, n_boot=600)
    assert abs(d2) < 1e-12 and lo2 <= 0 <= hi2


def test_cluster_diff_indep_and_did_recover_known_effects():
    rng = np.random.RandomState(2)
    sal = [_dict((rng.rand(300) < 0.80).astype(float)) for _ in range(3)]
    iid = [_dict((rng.rand(300) < 0.90).astype(float)) for _ in range(3)]
    d, lo, hi = stats.cluster_diff_indep(sal, iid, n_boot=500)
    assert -0.16 < d < -0.04 and lo < d < hi and hi < 0
    # arm B has the same gap between the sets: the difference in differences is about zero; arm A has a gap that is 0.1 larger
    a_sal = [_dict((rng.rand(300) < 0.70).astype(float)) for _ in range(3)]
    a_iid = [_dict((rng.rand(300) < 0.90).astype(float)) for _ in range(3)]
    est, lo2, hi2 = stats.cluster_did(a_sal, a_iid, sal, iid, n_boot=500)
    assert -0.2 < est < -0.02 and lo2 < est < hi2
