import os

import numpy as np
import pytest

from lychee import lang
from lychee.lang import FruitView
from lychee.modular import F_PX, choose_detection, depth_from_radius, parse_command, snap, views_from_detections
from lychee.select import IMG_HW


def test_parser_inverts_the_generator_for_every_template():
    n = 0
    for form in ("A", "B"):
        for spec in lang.all_specs():
            for variant in range(4):
                for word in range(2):
                    text = lang.render(spec, form=form, variant=variant, word=word)
                    assert parse_command(text) == spec, text
                    n += 1
    assert n > 300
    assert parse_command("pick the ripe lychee on the left without touching the green fruits") == lang.Spec("mat_side", 2, side="left")
    assert parse_command("dance") is None


def test_depth_from_radius_inverts_the_projection():
    for depth in (0.5, 0.7, 0.9):
        rad = F_PX * np.tan(np.arcsin(0.03 / depth))
        assert abs(depth_from_radius(rad) - depth) < 1e-6


def _table(rng, n):
    """A random fruit table in pixel space: detections with exact attributes."""
    x = np.sort(rng.uniform(20, IMG_HW[1] - 20, n))
    rng.shuffle(x)
    depth = rng.uniform(0.5, 0.9, n)
    return dict(x=x, y=rng.uniform(30, 120, n), rad=F_PX * np.tan(np.arcsin(0.03 / depth)), mat=rng.integers(0, 3, n), vis=rng.uniform(0.4, 1.0, n), score=rng.uniform(0.5, 1, n)), depth


def test_rules_on_a_perfect_table_reproduce_the_label_rules():
    """Parser + rules on the exact (pixel-space) fruit table give the target sets of lang.target_set on the exact FruitViews, for every valid command of random scenes."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        det, depth = _table(rng, int(rng.integers(3, 10)))
        views = [FruitView(j, int(det["mat"][j]), float(det["x"][j]) / IMG_HW[1], float(depth[j]), float(det["vis"][j])) for j in range(len(depth))]
        for spec, targets in lang.valid_specs_sets(views):
            text = lang.render(spec, form="A", variant=1, word=1)
            j = choose_detection(det, parse_command(text))
            assert j is not None and j in targets, (text, targets, j)
            checked += 1
    assert checked > 100


def test_margin_free_rules_also_reproduce_the_labels_when_the_margins_hold():
    """Without the label margins the rules still select a target of every valid command on a perfect table (the labels exist only where the margins hold)."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(60):
        det, depth = _table(rng, int(rng.integers(3, 10)))
        views = [FruitView(j, int(det["mat"][j]), float(det["x"][j]) / IMG_HW[1], float(depth[j]), float(det["vis"][j])) for j in range(len(depth))]
        for spec, targets in lang.valid_specs_sets(views):
            j = choose_detection(det, parse_command(lang.render(spec, form="A", variant=0, word=0)), strict=False)
            assert j is not None and j in targets, (spec, targets, j)
            checked += 1
    assert checked > 100


def test_margin_free_rules_answer_where_the_margins_leave_no_target():
    """Two ripe fruit at almost the same depth (1 cm apart, below the 3 cm margin): the strict rules find no nearest fruit, the margin-free rules pick the nearer one."""
    depth = np.array([0.700, 0.710, 0.80])
    det = dict(x=np.array([60.0, 120.0, 200.0]), y=np.array([50.0, 50.0, 50.0]), rad=F_PX * np.tan(np.arcsin(0.03 / depth)), mat=np.array([2, 2, 0]), vis=np.array([1.0, 1.0, 1.0]),
               score=np.array([0.9, 0.9, 0.9]))
    spec = lang.Spec("mat_depth", 2, depth="near")
    assert choose_detection(det, spec, strict=True) is None
    assert choose_detection(det, spec, strict=False) == 0
    assert choose_detection(det, lang.Spec("mat_depth", 2, depth="far"), strict=False) == 1
    assert choose_detection(det, lang.Spec("mat_side", 2, side="right"), strict=False) == 1
    assert choose_detection(det, lang.Spec("mat_ordinal", 2, side="left", k=3), strict=False) is None      # only two ripe detections


def test_snap_uses_the_nearest_true_centre_within_the_radius():
    det = dict(x=np.array([100.0, 200.0]), y=np.array([50.0, 60.0]), rad=np.array([10.0, 10.0]), mat=np.array([2, 2]), vis=np.array([1.0, 1.0]), score=np.array([1.0, 1.0]))
    fxy = np.array([[104.0, 52.0], [230.0, 60.0], [0.0, 0.0]])
    valid = np.array([True, True, False])
    assert snap(det, 0, fxy, valid, 20.0) == 0
    assert snap(det, 1, fxy, valid, 20.0) == -1                       # 30 px away
    assert snap(det, 1, fxy, valid, 40.0) == 1                        # a larger radius accepts it
    assert snap(det, None, fxy, valid, 20.0) == -1


@pytest.mark.skipif(not os.path.exists("D:/lychee_data/select_eval/iid_600/meta.npz"), reason="rendered evaluation set not available")
def test_oracle_pipeline_reproduces_the_benchmark_labels_on_rendered_scenes():
    d = np.load("D:/lychee_data/select_eval/iid_600/meta.npz")
    from lychee.detector import detections_from_table
    n = ok = 0
    for i in np.flatnonzero(d["ok"])[:200]:
        det = detections_from_table(d["fxy"][i], d["frad"][i], d["fmat"][i], d["fvis"][i], d["fvalid"][i])
        for k in range(int(d["ncmd"][i])):
            j = choose_detection(det, parse_command(d["text"][i, k]))
            sel = snap(det, j, d["fxy"][i], d["fvalid"][i], 20.0)
            n += 1; ok += bool(d["tmask"][i, k, sel]) if sel >= 0 else 0
    assert ok / n > 0.97, ok / n
