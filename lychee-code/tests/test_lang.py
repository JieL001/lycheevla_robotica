import re
import numpy as np
import pytest

from lychee.lang import (FruitView, Spec, resolve, valid_specs, counterfactuals, counterfactual_pairs,
                         render, MAT_WORDS, FRAMES, FIELDS, target_set, counterfactual_pairs_sets, valid_specs_sets)
from lychee.scene import sample_fruits


def F(idx, mat, u, depth=0.7, vis=1.0):
    return FruitView(idx, mat, u, depth, vis)


SCENE = [F(0, 2, 0.15), F(1, 2, 0.50, depth=0.55), F(2, 2, 0.85, depth=0.80),
         F(3, 0, 0.30), F(4, 1, 0.70)]


def test_mat_requires_unique_maturity():
    assert resolve(SCENE, Spec("mat", 0)) == 3          # single green fruit
    assert resolve(SCENE, Spec("mat", 1)) == 4
    assert resolve(SCENE, Spec("mat", 2)) is None       # three ripe fruits -> ambiguous


def test_side_depth_ordinal():
    assert resolve(SCENE, Spec("mat_side", 2, side="left")) == 0
    assert resolve(SCENE, Spec("mat_side", 2, side="right")) == 2
    assert resolve(SCENE, Spec("mat_depth", 2, depth="near")) == 1
    assert resolve(SCENE, Spec("mat_depth", 2, depth="far")) == 2
    assert resolve(SCENE, Spec("mat_ordinal", 2, side="left", k=2)) == 1
    assert resolve(SCENE, Spec("mat_ordinal", 2, side="right", k=3)) == 0


def test_degenerate_side_needs_two_candidates():
    # "the green lychee on the left" is meaningless when there is a single green fruit
    assert resolve(SCENE, Spec("mat_side", 0, side="left")) is None
    assert resolve(SCENE, Spec("mat_ordinal", 2, side="left", k=3)) == 2


def test_margin_makes_close_pairs_ambiguous():
    close = [F(0, 2, 0.40), F(1, 2, 0.42), F(2, 0, 0.9)]
    assert resolve(close, Spec("mat_side", 2, side="left"), u_margin=0.05) is None
    assert resolve(close, Spec("mat_side", 2, side="left"), u_margin=0.01) == 0


def test_min_visibility_filters_unpickable_targets():
    occ = [F(0, 2, 0.2, vis=0.05), F(1, 2, 0.8, vis=0.9)]
    assert resolve(occ, Spec("mat_side", 2, side="left"), min_vis=0.0) == 0
    assert resolve(occ, Spec("mat_side", 2, side="left"), min_vis=0.2) is None


def test_counterfactuals_switch_target_edit_one_field_and_stay_unique():
    for s, t in valid_specs(SCENE):
        for s2, t2, field in counterfactuals(SCENE, s):
            assert t2 != t and t2 is not None
            diff = [f for f in FIELDS[s.family] if getattr(s, f) != getattr(s2, f)]
            assert diff == [field]
            assert s.family == s2.family


def test_pairs_exist_in_example_scene():
    pairs = counterfactual_pairs(SCENE)
    assert any(f == "side" for *_, f in pairs)
    assert any(f == "mat" for *_, f in pairs)


def test_language_blind_policy_cannot_exceed_half_on_pairs():
    rng = np.random.default_rng(0)
    n_pairs = 0
    for _ in range(200):
        fruits = sample_fruits(rng, int(rng.integers(3, 11)))
        for s, t, s2, t2, _ in counterfactual_pairs(fruits):
            n_pairs += 1
            assert t != t2      # a fixed answer can satisfy at most one of the two commands
    assert n_pairs > 100


def test_surface_forms_A_B_share_no_maturity_word():
    words = lambda form: {w for v in MAT_WORDS[form].values() for w in v}
    assert words("A").isdisjoint(words("B"))


def test_render_has_no_unfilled_slots_and_is_deterministic():
    for form in "AB":
        for fam, frames in FRAMES[form].items():
            for v in range(len(frames)):
                spec = {"mat": Spec("mat", 2), "mat_side": Spec("mat_side", 2, side="left"),
                        "mat_depth": Spec("mat_depth", 1, depth="far"),
                        "mat_ordinal": Spec("mat_ordinal", 0, side="right", k=2),
                        "mat_any": Spec("mat_any", 2)}[fam]
                a, b = render(spec, form=form, variant=v), render(spec, form=form, variant=v)
                assert a == b and not re.search(r"[{}]", a) and "  " not in a


def test_avoid_clause():
    assert render(Spec("mat", 2), avoid=0).endswith("without touching the unripe fruits")


def test_any_of_family_target_sets():
    assert target_set(SCENE, Spec("mat_any", 2)) == (0, 1, 2)          # three ripe fruits, all acceptable
    assert target_set(SCENE, Spec("mat_any", 0)) == (3,)
    only_ripe = [F(0, 2, 0.2), F(1, 2, 0.6)]
    assert target_set(only_ripe, Spec("mat_any", 2)) == ()             # no distractor of another maturity: trivial
    low = [F(0, 2, 0.2, vis=0.1), F(1, 2, 0.6, vis=0.9), F(2, 0, 0.5)]
    assert target_set(low, Spec("mat_any", 2), min_vis=0.3) == (1,)    # the hidden ripe fruit is not a valid target
    assert target_set(low, Spec("mat_any", 2), min_vis=0.0, max_vis=0.5) == (0,)


def test_unique_families_have_singleton_sets_and_match_resolve():
    for s_, t in valid_specs(SCENE):
        assert target_set(SCENE, s_) == (t,)


def test_set_pairs_are_disjoint_and_differ_in_one_field():
    pairs = counterfactual_pairs_sets(SCENE)
    assert any(s1.family == "mat_any" for s1, *_ in pairs)
    for s1, T1, s2, T2, field in pairs:
        assert T1 and T2 and not set(T1) & set(T2)
        assert [f for f in FIELDS[s1.family] if getattr(s1, f) != getattr(s2, f)] == [field]


def test_any_of_rules_v10_and_v11():
    """v1.0: the target range of the split decides which fruit of the maturity are targets (a well-visible ripe fruit is not one on the occlusion split); v1.1: every visible-enough fruit of the maturity is a target and
    the command exists only if none is above the range; where the rules both give a target set (unambiguous scenes) the sets are equal; with max_vis = 1 the two rules coincide."""
    mixed = [F(0, 2, 0.2, vis=0.3), F(1, 2, 0.6, vis=0.9), F(2, 0, 0.5, vis=0.3)]
    assert target_set(mixed, Spec("mat_any", 2), min_vis=0.2, max_vis=0.4) == (0,)
    assert target_set(mixed, Spec("mat_any", 2), min_vis=0.2, max_vis=0.4, any_rule="v1.1") == ()
    occluded = [F(0, 2, 0.2, vis=0.3), F(1, 2, 0.6, vis=0.35), F(2, 0, 0.5, vis=0.9), F(3, 2, 0.4, vis=0.1)]
    for rule in ("v1.0", "v1.1"):
        assert target_set(occluded, Spec("mat_any", 2), min_vis=0.2, max_vis=0.4, any_rule=rule) == (0, 1)       # the 0.1-visible ripe fruit is a target under neither rule
    assert target_set(mixed, Spec("mat_any", 2), min_vis=0.2, any_rule="v1.1") == (0, 1)                        # no upper bound: the rules coincide
    assert target_set(mixed, Spec("mat_any", 2), min_vis=0.2) == (0, 1)
    for sc in (SCENE, mixed, occluded):
        for spec in (Spec("mat_any", m) for m in range(3)):
            assert target_set(sc, spec, min_vis=0.2, any_rule="v1.1") == target_set(sc, spec, min_vis=0.2)
    only_ripe = [F(0, 2, 0.2, vis=0.3), F(1, 2, 0.6, vis=0.35)]
    assert target_set(only_ripe, Spec("mat_any", 2), min_vis=0.2, max_vis=0.4, any_rule="v1.1") == ()          # still needs a distractor of another maturity
    with pytest.raises(ValueError):
        target_set(SCENE, Spec("mat_any", 2), any_rule="v2.0")
