import numpy as np
import pytest

from lychee import evalkit as ek


def rec(index, which, targets, first_detached, n_fruit=5, grasped=None, success=None, touched=False, family="mat_side"):
    first_grasped = first_detached if grasped is None else grasped
    correct = first_detached in targets
    return dict(split="iid", index=index, which=which, family=family, n_fruit=n_fruit, targets=targets,
                first_detached=first_detached, first_grasped=first_grasped, target_correct=correct,
                success=correct if success is None else success, wrong_target=first_detached >= 0 and not correct,
                touched_nontarget=touched, approach_correct=correct)


def test_wilson_is_not_degenerate_at_the_boundaries():
    p, lo, hi = ek.wilson_ci(0, 60)
    assert p == 0 and lo == 0 and 0.055 < hi < 0.065              # exact upper bound is about 6%
    p, lo, hi = ek.wilson_ci(60, 60)
    assert p == 1 and 0.935 < lo < 0.945 and hi > 1 - 1e-9
    assert ek.wilson_ci(30, 60)[1] < 0.5 < ek.wilson_ci(30, 60)[2]
    assert np.isnan(ek.wilson_ci(0, 0)[0])


def test_language_blind_policy_has_pta_zero_and_full_collapse():
    recs = []
    for i in range(20):                                             # picks fruit 0 for both commands, targets are 0 / 1
        recs += [rec(i, "plus", [0], 0), rec(i, "minus", [1], 0)]
    s = ek.summarize(recs)["all"]
    assert s["n_pairs"] == 20 and s["PTA_sel"][0] == 0.0
    assert s["TSA"][0] == 0.5                                       # right for exactly one member of every pair
    assert s["same_first_fruit"][0] == 1.0
    assert s["out_correct"][0] == 0.5 and s["out_wrong"][0] == 0.5 and s["out_none"][0] == 0.0


def test_perfect_policy_and_three_way_outcome():
    recs = []
    for i in range(10):
        recs += [rec(i, "plus", [0], 0), rec(i, "minus", [1], 1)]
    for i in range(10, 15):                                         # nothing detaches for the minus command
        recs += [rec(i, "plus", [0], 0), rec(i, "minus", [1], -1, grasped=1)]
    s = ek.summarize(recs)["all"]
    assert abs(s["PTA_sel"][0] - 10 / 15) < 1e-9
    assert abs(s["out_none"][0] - 5 / 30) < 1e-9
    assert abs(s["PTA_grasp"][0] - 1.0) < 1e-9                      # it did close on the right fruit for both commands
    assert abs(s["grasp_rate"][0] - 1.0) < 1e-9


def test_clean_success_counts_contact():
    recs = [rec(0, "plus", [0], 0, touched=True), rec(0, "minus", [1], 1)]
    s = ek.summarize(recs)["all"]
    assert s["success"][0] == 1.0 and s["clean_success"][0] == 0.5


def test_chance_pta_of_a_uniform_selector():
    # 5 fruit, both target sets of size 1: chance = (1/5)(1/5); any-of with 2 targets vs 1 target: (2/5)(1/5)
    a = [rec(0, "plus", [0], 0), rec(0, "minus", [1], 1)]
    b = [rec(1, "plus", [0, 2], 0), rec(1, "minus", [1], 1)]
    assert abs(ek.summarize(a)["all"]["chance_PTA"] - 0.04) < 1e-12
    assert abs(ek.summarize(a + b)["all"]["chance_PTA"] - (0.04 + 0.08) / 2) < 1e-12


def test_by_family_and_macro_average():
    recs = []
    for i in range(6):
        recs += [rec(i, "plus", [0], 0, family="mat_side"), rec(i, "minus", [1], 1, family="mat_side")]     # PTA 1
    for i in range(6, 8):
        recs += [rec(i, "plus", [0], 0, family="mat_any"), rec(i, "minus", [1], 0, family="mat_any")]      # PTA 0
    by = ek.summarize(recs, by="family")
    assert by["mat_side"]["PTA_sel"][0] == 1.0 and by["mat_any"]["PTA_sel"][0] == 0.0
    macro, lo, hi, sizes = ek.macro_average(recs)
    assert abs(macro - 0.5) < 1e-9 and sizes == {"mat_side": 6, "mat_any": 2} and lo <= macro <= hi
    assert abs(ek.summarize(recs)["all"]["PTA_sel"][0] - 6 / 8) < 1e-9        # the pooled number depends on the mix, the macro one does not


def test_unpaired_records_are_ignored_by_the_pair_table():
    assert ek.pair_table([rec(0, "plus", [0], 0)]) == {}
