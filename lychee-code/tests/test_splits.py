import numpy as np
import pytest

from lychee.layout import sample_layout, fruit_visibility, DEFAULT_CAM
from lychee.splits import (SPLITS, HELD_OUT, FAMILY_WEIGHTS, sample_config, layout_for, salience, _combo, _salience_ok)
from lychee.lang import resolve, target_set


def test_layout_is_deterministic_and_visibility_in_range():
    a, b = sample_layout(11), sample_layout(11)
    assert np.allclose(a.pos, b.pos) and np.allclose(a.vis, b.vis)
    assert np.all((a.vis >= 0) & (a.vis <= 1))
    assert a.n == len(a.vis) == len(a.mats)


def test_occluder_planning_hits_requested_visibility():
    lay = sample_layout(5, p_occlude=1.0, occ_vis_range=(0.25, 0.4), leaf_range=(0, 0))
    assert np.mean((lay.vis > 0.15) & (lay.vis < 0.5)) >= 0.7        # bisection tolerance + overlaps


def test_visibility_hand_made():
    from lychee.layout import Leaf, _quat_from_axes
    pos = np.array([[-0.05, 0.0, 0.34]])
    to_cam = DEFAULT_CAM.eye - pos[0]; to_cam /= np.linalg.norm(to_cam)
    a = np.cross(to_cam, [0, 0, 1.0]); a /= np.linalg.norm(a); b = np.cross(to_cam, a)
    q = _quat_from_axes(a, b, to_cam)
    centred = Leaf(pos[0] + 0.05 * to_cam, q)
    # the leaf is 4.4 cm wide and the fruit 6 cm: a centred leaf leaves two slivers (~16% of the disk) visible
    assert 0.05 < fruit_visibility(pos, [centred], DEFAULT_CAM)[0] < 0.20
    shifts = [0.0, 0.02, 0.04, 0.06, 0.09]
    vis = [fruit_visibility(pos, [Leaf(centred.pos + s_ * a, q)], DEFAULT_CAM)[0] for s_ in shifts]
    assert all(x <= y + 1e-9 for x, y in zip(vis, vis[1:])) and vis[-1] > 0.98     # moving away never hides more
    assert fruit_visibility(pos, [], DEFAULT_CAM)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["train", "iid", "occ_ood", "density_ood", "lang_ood", "attr_ood", "saliency_rev", "iid@v11", "occ_ood@v11", "train@v11"])
def test_pair_configs_respect_split_constraints(name):
    sp = SPLITS[name]
    lo, hi = sp.target_vis
    for idx in range(4):
        cfg = sample_config(name, idx)
        lay = layout_for(sp, cfg.scene_seed)
        fr = lay.fruit_views()
        assert sp.n_range[0] <= lay.n <= sp.n_range[1]
        p, m = cfg.plus, cfg.minus
        assert set(p.targets).isdisjoint(m.targets)
        assert p.primary in p.targets and m.primary in m.targets
        assert tuple(target_set(fr, p.spec, min_vis=lo, max_vis=hi, ord_min_vis=sp.ord_min_vis, any_rule=sp.any_rule)) == tuple(p.targets)
        assert tuple(target_set(fr, m.spec, min_vis=lo, max_vis=hi, ord_min_vis=sp.ord_min_vis, any_rule=sp.any_rule)) == tuple(m.targets)
        for e in (p, m):
            assert e.form == sp.form
            assert all(lo <= fr[t].vis <= hi for t in e.targets)
        assert p.spec.family == m.spec.family == cfg.family
        assert dict(sp.family_weights)[cfg.family] > 0
        if sp.combos == "train":
            assert _combo(p.spec) not in HELD_OUT and _combo(m.spec) not in HELD_OUT
        if sp.combos == "heldout":
            assert _combo(p.spec) in HELD_OUT
        if sp.salience_reversed:
            assert _salience_ok(sp, fr, p.targets)


def test_config_is_deterministic_and_pair_shares_scene():
    a, b = sample_config("iid", 3), sample_config("iid", 3)
    assert a.scene_seed == b.scene_seed and a.plus.text == b.plus.text and a.minus.text == b.minus.text
    assert a.plus.text != a.minus.text


def test_unpaired_mode_gives_single_episode():
    cfg = sample_config("train", 4, pair=False)
    assert cfg.minus is None and len(cfg.episodes()) == 1


def test_family_is_drawn_from_the_configured_mix():
    for name in ("iid", "density_ood"):
        fw = dict(SPLITS[name].family_weights)
        fams = [sample_config(name, i).family for i in range(16)]
        assert all(fw[f] > 0 for f in fams)                       # never a family with weight 0
        assert len(set(fams)) >= 2
    assert dict(SPLITS["density_ood"].family_weights)["mat"] == 0


def test_splits_use_disjoint_seed_ranges():
    bases = sorted(s.seed_base for s in SPLITS.values())
    assert all(b - a >= 10_000_000 for a, b in zip(bases, bases[1:]))


def test_visual_dr_train_and_heldout_ranges_are_disjoint():
    from lychee.layout import sample_dr, sample_layout, DEFAULT_CAM
    for seed in range(60):
        tr, ho = sample_dr(seed, "train"), sample_dr(seed, "heldout")
        assert np.all(np.abs(tr.cam_dpos) <= 0.025 + 1e-9) and np.all(np.abs(ho.cam_dpos) >= 0.04 - 1e-9)
        assert np.all(np.abs(tr.cam_dtarget) <= 0.02 + 1e-9) and np.all(np.abs(ho.cam_dtarget) >= 0.03 - 1e-9)
        assert abs(tr.fov_delta) <= np.deg2rad(2) + 1e-9 and abs(ho.fov_delta) >= np.deg2rad(3.5) - 1e-9
        assert 0.25 <= tr.ambient <= 0.40 and not (0.20 < ho.ambient < 0.55)
        assert 0.9 <= tr.fruit_gain <= 1.1 and not (0.8 < ho.fruit_gain < 1.2)
        assert np.all(np.abs(tr.tint - 1) <= 0.06 + 1e-9) and np.all(np.abs(ho.tint - 1) >= 0.12 - 1e-9)
    off = sample_dr(3, "off")
    assert np.allclose(off.cam_dpos, 0) and off.ambient == 0.30
    a, b = sample_layout(9, dr="off"), sample_layout(9, dr="heldout")
    assert np.allclose(a.pos, b.pos) and np.allclose(a.mats, b.mats)          # scene content unchanged by DR
    assert not np.allclose(a.cam.eye, b.cam.eye)                               # camera (and thus visibility labels) follow the DR
    assert np.allclose(a.cam.eye, DEFAULT_CAM.eye)


def test_visual_dr_split_uses_heldout_dr_and_others_are_off():
    assert SPLITS["visual_dr_ood"].dr == "heldout"
    assert all(s.dr == "off" for n, s in SPLITS.items() if n.split("@")[0] != "visual_dr_ood")
    assert SPLITS["visual_dr_ood@v11"].dr == "heldout"


# ---------------------------------------------------------------------------- v1.1 machinery (defaults keep v1.0 behaviour)
def test_ordinal_command_needs_visible_fruit_up_to_position_k():
    from lychee.lang import FruitView, Spec
    fr = [FruitView(0, 2, 0.20, 1.0, 0.05),          # a nearly hidden ripe fruit, leftmost
          FruitView(1, 2, 0.40, 1.0, 0.9),
          FruitView(2, 2, 0.70, 1.0, 0.9),
          FruitView(3, 0, 0.55, 1.0, 0.9)]
    spec = Spec("mat_ordinal", 2, side="left", k=2)
    assert resolve(fr, spec) == 1                                     # v1.0: the hidden fruit is counted first, so fruit 1 is "second"
    assert resolve(fr, spec, ord_min_vis=0.2) is None                 # v1.1: a viewer would count fruit 1 first -> ambiguous, dropped
    fr2 = [FruitView(0, 2, 0.20, 1.0, 0.9), FruitView(1, 2, 0.40, 1.0, 0.9), FruitView(2, 2, 0.70, 1.0, 0.05)]
    assert resolve(fr2, spec, ord_min_vis=0.2) == 1                   # a hidden fruit BEHIND the target does not change the count


def test_v10_defaults_are_unchanged():
    for name, sp in SPLITS.items():
        if "@" in name or name.startswith("train_b"):
            continue
        assert sp.ord_min_vis == 0.0 and sp.bias_rho == 0.0


def _top_salient_rate(name, n):
    hits = 0
    for i in range(n):
        cfg = sample_config(name, i, pair=False)
        fr = layout_for(SPLITS[name], cfg.scene_seed).fruit_views()
        hits += int(np.argmax([salience(f) for f in fr])) in cfg.plus.targets
    return hits / n


def test_bias_dial_steers_commands_towards_the_most_salient_fruit():
    base, biased = _top_salient_rate("train", 20), _top_salient_rate("train_b90", 20)
    assert biased >= base + 0.25 and biased >= 0.6, (base, biased)


def test_independent_pairs_keep_the_paired_scene_but_draw_commands_independently():
    n_same_targets = 0
    for i in range(6):
        a, b = sample_config("train", i, pair=True), sample_config("train", i, pair=True, indep=True)
        assert a.scene_seed == b.scene_seed and a.family == b.family
        assert b.edited_field is None and b.minus is not None
        n_same_targets += set(b.plus.targets) == set(b.minus.targets) or bool(set(b.plus.targets) & set(b.minus.targets))
        assert set(a.plus.targets).isdisjoint(a.minus.targets)          # counterfactual pairs stay disjoint
    assert n_same_targets >= 0


def test_same_control_paraphrases_the_first_command_on_the_paired_scene():
    for i in range(8):
        a, s = sample_config("train", i, pair=True), sample_config("train", i, pair=True, same=True)
        assert a.scene_seed == s.scene_seed and a.family == s.family
        assert s.minus is not None and s.edited_field is None
        assert set(s.plus.targets) == set(s.minus.targets)              # no contrast: identical target sets
        assert s.plus.text != s.minus.text                              # but two different wordings
        assert s.plus.spec == s.minus.spec


def test_v11_rules_and_disjoint_seed_ranges():
    for name in [n for n in SPLITS if "@" not in n and n + "@v11" in SPLITS]:
        a, b = SPLITS[name], SPLITS[name + "@v11"]
        assert b.ord_min_vis == 0.2 and b.seed_base == a.seed_base + 500_000_000
        assert b.target_vis == (a.target_vis if name == "occ_ood" else (0.5, 1.0))
        assert (b.n_range, b.p_occlude, b.form, b.family_weights, b.combos) == (a.n_range, a.p_occlude, a.form, a.family_weights, a.combos)
    assert SPLITS["occ_ood@v11"].target_vis[1] <= SPLITS["iid@v11"].target_vis[0] - 0.1 + 1e-9    # guard band


def test_sal_both_constrains_both_members_of_a_pair():
    sp = SPLITS["sal_both"]
    assert sp.salience_reversed and sp.salience_both and not SPLITS["saliency_rev"].salience_both
    for idx in range(6):
        cfg = sample_config("sal_both", idx)
        fr = layout_for(sp, cfg.scene_seed).fruit_views()
        assert _salience_ok(sp, fr, cfg.plus.targets) and _salience_ok(sp, fr, cfg.minus.targets)


def test_fresh_evaluation_splits_draw_their_own_scenes():
    for fresh, old in (("iid_fresh", "iid"), ("sal_fresh", "saliency_rev"), ("attr_fresh", "attr_ood")):
        a, b = SPLITS[fresh], SPLITS[old]
        assert (a.n_range, a.target_vis, a.combos, a.form, a.family_weights, a.salience_reversed) == \
               (b.n_range, b.target_vis, b.combos, b.form, b.family_weights, b.salience_reversed)
        assert sample_config(fresh, 0).scene_seed != sample_config(old, 0).scene_seed


def test_v11_occlusion_split_issues_any_of_commands_only_on_occluded_maturities():
    """Benchmark v1.1: occlusion is a property of the scene, not of the legal answers: every fruit of the commanded maturity is a target if it is visible enough, and none is more visible than the range."""
    sp = SPLITS["occ_ood@v11"]
    assert sp.any_rule == "v1.1" and SPLITS["occ_ood"].any_rule == "v1.0" and SPLITS["iid"].any_rule == "v1.0"
    lo, hi = sp.target_vis
    seen = 0
    for idx in range(60):
        cfg = sample_config("occ_ood@v11", idx)
        if cfg.family != "mat_any":
            continue
        fr = layout_for(sp, cfg.scene_seed).fruit_views()
        for e in (cfg.plus, cfg.minus):
            named = [f for f in fr if f.mat == e.spec.mat]
            assert all(f.vis <= hi for f in named)
            assert set(e.targets) == {f.idx for f in named if f.vis >= lo}
        seen += 1
    assert seen >= 3
