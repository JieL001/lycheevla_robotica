"""Modular baseline for command-conditioned target selection: perception network (fruit detector, lychee/detector.py) + rule-based command parser + the benchmark's own
target rules (lychee/lang.py) applied to the PREDICTED fruit table.  No learning of language, no privileged simulator state at test time: the detector sees the same cropped
third-person image as the selection networks and the depth is obtained from the predicted apparent radius with the known camera and the known fruit radius.

    text --parse_command--> Spec            image --FruitDetector--> detections (x, y, radius, maturity, visibility, score)
    Spec + predicted fruit views --lang.target_set--> target set among the detections --> the detection to point at (any-of: the most visible one)
The pointing position is then handled exactly like the output of a selection network (the nearest true fruit centre within the snap radius; the same scripted expert executes).
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from . import lang
from .lang import FruitView, Spec
from .layout import DEFAULT_CAM
from .select import IMG_HW

FRUIT_RADIUS = 0.03                                            # m, the benchmark's fruit radius (a constant of the scene generator)
F_PX = (IMG_HW[1] / 2) / math.tan(DEFAULT_CAM.fov / 2)         # focal length in pixels of the fixed third-person camera
_SUFFIX = re.compile(r"\s+without touching the [\w\- ]+ fruits?$")
_LEX: Optional[dict] = None


def depth_from_radius(rad_px: float) -> float:
    """Depth (m) of a fruit of the known radius from its apparent radius in pixels (inverse of the projection used to render/label: rad = F_PX * tan(asin(R / depth)))."""
    return FRUIT_RADIUS / math.sin(math.atan(max(rad_px, 1e-3) / F_PX))


def _lexicon() -> dict:
    """Every command text that the generator can emit (both surface forms, all frames and maturity synonyms) -> its Spec.  This is the rule-based parser: the templates are closed."""
    global _LEX
    if _LEX is None:
        _LEX = {}
        for form in ("A", "B"):
            for spec in lang.all_specs():
                for variant in range(4):
                    for word in range(2):
                        _LEX[lang.render(spec, form=form, variant=variant, word=word).lower()] = spec
    return _LEX


def parse_command(text: str) -> Optional[Spec]:
    """Spec of a command text, or None if it matches no template (then the modular baseline cannot answer)."""
    return _lexicon().get(_SUFFIX.sub("", " ".join(str(text).lower().split())))


def views_from_detections(det) -> list:
    """FruitView list of the detections (index = position in the detection list): u from the x coordinate, depth from the apparent radius, maturity and visibility as predicted."""
    return [FruitView(idx=j, mat=int(det["mat"][j]), u=float(det["x"][j]) / IMG_HW[1], depth=depth_from_radius(float(det["rad"][j])), vis=float(det["vis"][j]))
            for j in range(len(det["x"]))]


def choose_detection(det, spec: Optional[Spec], strict: bool = True) -> Optional[int]:
    """Index (in the detection list) of the detection that the rules select for ``spec``, or None.
    strict=True: the rules of the benchmark (margins of 5 % of the image width and 3 cm of depth between the target and the runner-up, an unambiguous ordering up to the k-th fruit; ordinals
    count the detected fruit; any-of: the most visible detection of the maturity).  The margins make the LABELS unambiguous; applied to predicted positions and depths they turn small
    perception errors into commands without a target.  strict=False: the same rules without the margins, as a solver would use them: the leftmost, rightmost, nearest or farthest detection
    of the maturity, the k-th detection counted from the side, the highest-scoring detection for a unique maturity, the most visible one for any-of."""
    if spec is None or len(det["x"]) == 0:
        return None
    if not strict:
        cands = [j for j in range(len(det["x"])) if int(det["mat"][j]) == spec.mat]
        if not cands:
            return None
        if spec.family == "mat_any":
            return int(max(cands, key=lambda j: (det["vis"][j], det["score"][j])))
        if spec.family == "mat":
            return int(max(cands, key=lambda j: det["score"][j]))
        if spec.family == "mat_side":
            return int((max if spec.side == "right" else min)(cands, key=lambda j: det["x"][j]))
        if spec.family == "mat_depth":
            return int((min if spec.depth == "near" else max)(cands, key=lambda j: depth_from_radius(float(det["rad"][j]))))
        if spec.family == "mat_ordinal":
            order = sorted(cands, key=lambda j: det["x"][j], reverse=(spec.side == "right"))
            return int(order[spec.k - 1]) if spec.k is not None and len(order) >= spec.k else None
        return None
    views = views_from_detections(det)
    targets = lang.target_set(views, spec)
    if not targets:
        return None
    if len(targets) == 1:
        return int(targets[0])
    return int(max(targets, key=lambda j: (det["vis"][j], det["score"][j])))


def snap(det, j: Optional[int], fxy, fvalid, radius: float) -> int:
    """Pointing position of detection j -> index of the nearest true fruit centre within ``radius`` px (the interface of all selectors), or -1."""
    if j is None:
        return -1
    d = np.hypot(np.asarray(fxy)[:, 0] - det["x"][j], np.asarray(fxy)[:, 1] - det["y"][j])
    d = np.where(np.asarray(fvalid, bool), d, np.inf)
    i = int(np.argmin(d))
    return i if d[i] <= radius else -1
