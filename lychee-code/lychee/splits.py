"""Splits and episode-config sampling for LycheeHarvest (numpy only, no simulator).

An *episode config* fully specifies one episode: the scene (via ``scene_seed`` + layout knobs), the
instruction (family/attributes/surface form) and its target set.  A *pair config* holds two of them on
the SAME scene whose commands differ in exactly one discriminative field and whose target sets are
disjoint.

The command family is drawn once per config from the split's ``family_weights``, so a split changes
only what it is meant to change -- e.g. density-OOD does not silently change the command mix.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .lang import FruitView, Spec, counterfactual_pairs_sets, target_set, valid_specs_sets, render
from .layout import sample_layout, Layout

FAMILY_WEIGHTS = {"mat": 0.05, "mat_side": 0.25, "mat_depth": 0.20, "mat_ordinal": 0.25, "mat_any": 0.25}
# (family, maturity) combinations never commanded in training; used by the attribute-OOD split
HELD_OUT = {("mat_side", 0), ("mat_depth", 1), ("mat_ordinal", 2), ("mat_any", 1)}


@dataclass(frozen=True)
class SplitConfig:
    name: str
    n_range: tuple = (3, 10)
    target_vis: tuple = (0.4, 1.0)      # allowed visibility of every commanded target
    combos: str = "train"               # "train" (no HELD_OUT) | "heldout" (positive is HELD_OUT) | "all"
    form: str = "A"                     # surface form of the command: "A" | "B"
    p_occlude: float = 0.35
    occ_vis_range: tuple = (0.2, 1.0)
    leaf_range: tuple = (8, 30)
    salience_reversed: bool = False     # the commanded target(s) must not be the most salient fruit(s)
    salience_both: bool = False         # with salience_reversed: the second member of a pair obeys the rule too (default: only the first)
    dr: str = "off"                     # visual domain randomisation: "off" | "train" | "heldout"
    ord_min_vis: float = 0.0            # ordinal commands: every fruit up to position k must be at least this visible
    any_rule: str = "v1.0"              # any-of commands: "v1.0" targets = fruit of the maturity inside the visibility range of the split; "v1.1" = every fruit at least min_vis visible, issued only if none is above the range
    bias_rho: float = 0.0               # unpaired data only: P(the commanded target is the most salient fruit) -- the "bias dial"
    seed_base: int = 0
    family_weights: tuple = tuple(FAMILY_WEIGHTS.items())


SPLITS = {
    "train": SplitConfig("train", seed_base=0),
    "val": SplitConfig("val", seed_base=10_000_000),
    "iid": SplitConfig("iid", seed_base=20_000_000),
    "occ_ood": SplitConfig("occ_ood", target_vis=(0.2, 0.4), p_occlude=0.75, occ_vis_range=(0.2, 0.45), seed_base=30_000_000),
    # at 11-16 fruits no scene has two maturity classes with exactly one fruit each, so `mat` pairs cannot exist
    "density_ood": SplitConfig("density_ood", n_range=(11, 16), leaf_range=(12, 34), seed_base=40_000_000,
                               family_weights=(("mat", 0.0), ("mat_side", 0.30), ("mat_depth", 0.20),
                                               ("mat_ordinal", 0.25), ("mat_any", 0.25))),
    "lang_ood": SplitConfig("lang_ood", form="B", seed_base=50_000_000),
    "attr_ood": SplitConfig("attr_ood", combos="heldout", seed_base=60_000_000),
    "saliency_rev": SplitConfig("saliency_rev", salience_reversed=True, seed_base=70_000_000),
    "visual_dr_ood": SplitConfig("visual_dr_ood", dr="heldout", seed_base=80_000_000),
}
# bias-dial training sets (same scene/command distribution as ``train`` except that the command is steered towards the most salient fruit); own seed ranges
SPLITS["train_b50"] = replace(SPLITS["train"], name="train_b50", bias_rho=0.5, seed_base=100_000_000)
SPLITS["train_b90"] = replace(SPLITS["train"], name="train_b90", bias_rho=0.9, seed_base=110_000_000)


# Benchmark version 1.1 (confirmatory runs).  Differences to v1.0: (i) an ordinal command needs every fruit of the commanded
# maturity up to position k to be at least ``ord_min_vis`` visible, so that counting the fruit a viewer can see gives the
# label; (ii) a guard band of 0.1 between the target visibility of the training/IID distribution (>= 0.5) and of the
# occlusion split (0.2-0.4); (iii) the any-of rule of ``lang.target_set`` (every fruit of the named maturity that is visible enough is a target; on the
# occlusion split the command is issued only if all fruit of that maturity are occluded).  Names carry the suffix "@v11" and use seed ranges disjoint from v1.0 (+500 M), so the
# exploratory pilot on v1.0 never overlaps with the confirmatory data.
V11_SEED_SHIFT = 500_000_000


def _v11(cfg: SplitConfig) -> SplitConfig:
    tv = cfg.target_vis if cfg.name == "occ_ood" else (0.5, 1.0)
    return replace(cfg, name=cfg.name + "@v11", ord_min_vis=0.2, target_vis=tv, any_rule="v1.1", seed_base=cfg.seed_base + V11_SEED_SHIFT)


for _name in list(SPLITS):
    SPLITS[_name + "@v11"] = _v11(SPLITS[_name])

# Evaluation sets added after the pilot (own seed ranges, benchmark v1.0 rules): fresh scenes for the IID, saliency-reversed and attribute
# splits, so that the headline comparisons can be re-checked on scenes that no design decision saw, and a saliency-reversed split in which
# BOTH commands of a pair avoid the most salient fruit (the original one constrains only the first member).
SPLITS["iid_fresh"] = replace(SPLITS["iid"], name="iid_fresh", seed_base=120_000_000)
SPLITS["sal_fresh"] = replace(SPLITS["saliency_rev"], name="sal_fresh", seed_base=130_000_000)
SPLITS["attr_fresh"] = replace(SPLITS["attr_ood"], name="attr_fresh", seed_base=140_000_000)
SPLITS["sal_both"] = replace(SPLITS["saliency_rev"], name="sal_both", salience_both=True, seed_base=150_000_000)


def _combo(spec: Spec):
    return (spec.family, spec.mat)


def _combo_ok(split: SplitConfig, spec: Spec, *, positive: bool) -> bool:
    held = _combo(spec) in HELD_OUT
    if split.combos == "train":
        return not held
    if split.combos == "heldout":
        return held if positive else True
    return True


def salience(f: FruitView) -> float:
    """Apparent size x visibility: visible projected area, up to a constant (1 / depth^2)."""
    return f.vis / (f.depth ** 2)


def _salience_ok(split: SplitConfig, fruits, targets: tuple) -> bool:
    """Saliency-reversed split.  Unique target: it lies in the less-salient half of the fruits.  Any-of set:
    the single most salient fruit is not part of it."""
    if not split.salience_reversed:
        return True
    s = np.array([salience(f) for f in fruits])
    if len(targets) == 1:
        return int((s < s[targets[0]]).sum()) < int(np.ceil(len(fruits) / 2))
    return int(np.argmax(s)) not in targets


@dataclass
class Episode:
    spec: Spec
    targets: tuple                  # every fruit index that counts as correct
    primary: int                    # the fruit the scripted expert goes for
    text: str
    form: str
    variant: int
    word: int

    @property
    def target(self) -> int:        # the unique-target case / the expert's target
        return self.primary


@dataclass
class PairConfig:
    split: str
    index: int
    scene_seed: int
    layout: dict                    # knobs to rebuild the layout from ``scene_seed``
    plus: Episode
    minus: Optional[Episode]
    edited_field: Optional[str]
    family: str

    def episodes(self):
        return [e for e in (self.plus, self.minus) if e is not None]


def layout_for(split: SplitConfig, scene_seed: int) -> Layout:
    return sample_layout(scene_seed, n_range=split.n_range, leaf_range=split.leaf_range,
                         p_occlude=split.p_occlude, occ_vis_range=split.occ_vis_range, dr=split.dr)


def _episode(spec: Spec, targets: tuple, fruits, form: str, rng) -> Episode:
    variant, word = int(rng.randint(2)), int(rng.randint(2))
    if len(targets) == 1:
        primary = targets[0]
    else:                                                     # any-of: prefer the more visible fruits
        w = np.array([max(fruits[t].vis, 1e-3) for t in targets])
        primary = int(targets[int(rng.choice(len(targets), p=w / w.sum()))])
    return Episode(spec, tuple(targets), primary, render(spec, form=form, variant=variant, word=word), form, variant, word)


def _paraphrase(spec: Spec, targets: tuple, fruits, form: str, rng, prev: Episode) -> Episode:
    """Another wording (verb frame and maturity synonym) of the same instruction: same fields, same target set."""
    e = prev
    for _ in range(12):
        e = _episode(spec, targets, fruits, form, rng)
        if e.text != prev.text:
            break
    return e


def sample_config(split: SplitConfig | str, index: int, *, pair: bool = True, indep: bool = False, same: bool = False,
                  max_tries: int = 400) -> PairConfig:
    """Deterministic in (split, index).  ``pair=False`` gives a single instruction (unpaired baseline data).
    ``pair=True, indep=True`` keeps the scene (and command family) a counterfactual pair would have used but draws the two
    commands independently from the scene's valid commands: the control that separates the effect of the contrast from
    that of showing two commands on one scene.  Independent draws still name disjoint fruit in about half of the scenes;
    ``same=True`` removes the contrast completely: the second command is a paraphrase of the first (identical target set)."""
    split = SPLITS[split] if isinstance(split, str) else split
    rng = np.random.RandomState(split.seed_base + index)
    fw = dict(split.family_weights)
    fams = [f for f in fw if fw[f] > 0 and (split.combos != "heldout" or any(fam == f for fam, _ in HELD_OUT))]
    probs = np.array([fw[f] for f in fams])
    fam = fams[int(rng.choice(len(fams), p=probs / probs.sum()))]        # drawn ONCE, so the mix is as configured
    lo, hi = split.target_vis
    knobs = dict(n_range=split.n_range, leaf_range=split.leaf_range, p_occlude=split.p_occlude,
                 occ_vis_range=split.occ_vis_range, dr=split.dr)
    for _ in range(max_tries):
        scene_seed = int(rng.randint(2 ** 31 - 1))
        fruits = layout_for(split, scene_seed).fruit_views()
        vkw = dict(min_vis=lo, max_vis=hi, ord_min_vis=split.ord_min_vis, any_rule=split.any_rule)
        if pair:
            cands = []
            for s1, T1, s2, T2, fld in counterfactual_pairs_sets(fruits, **vkw):
                if s1.family != fam:
                    continue
                for a, Ta, b, Tb in ((s1, T1, s2, T2), (s2, T2, s1, T1)):        # either member may be ``plus``
                    if _combo_ok(split, a, positive=True) and _combo_ok(split, b, positive=False) \
                            and _salience_ok(split, fruits, Ta) and (not split.salience_both or _salience_ok(split, fruits, Tb)):
                        cands.append((a, Ta, b, Tb, fld))
            if not cands:
                continue
            if indep or same:
                singles = [(s_, T_) for s_, T_ in valid_specs_sets(fruits, **vkw) if s_.family == fam
                           and _combo_ok(split, s_, positive=True) and _salience_ok(split, fruits, T_)]
                if same:
                    a, Ta = singles[int(rng.randint(len(singles)))]
                    e1 = _episode(a, Ta, fruits, split.form, rng)
                    return PairConfig(split.name, index, scene_seed, knobs, e1, _paraphrase(a, Ta, fruits, split.form, rng, e1), None, fam)
                (a, Ta), (b, Tb) = (singles[int(rng.randint(len(singles)))] for _ in range(2))
                return PairConfig(split.name, index, scene_seed, knobs, _episode(a, Ta, fruits, split.form, rng),
                                  _episode(b, Tb, fruits, split.form, rng), None, fam)
            a, Ta, b, Tb, fld = cands[int(rng.randint(len(cands)))]
            return PairConfig(split.name, index, scene_seed, knobs, _episode(a, Ta, fruits, split.form, rng),
                              _episode(b, Tb, fruits, split.form, rng), fld, fam)
        singles = [(s, T) for s, T in valid_specs_sets(fruits, **vkw) if s.family == fam
                   and _combo_ok(split, s, positive=True) and _salience_ok(split, fruits, T)]
        if not singles:
            continue
        if split.bias_rho > 0 and rng.rand() < split.bias_rho:          # steer the command towards the most salient fruit
            top = int(np.argmax([salience(f) for f in fruits]))
            pool = [(s_, T_) for s_, T_ in valid_specs_sets(fruits, **vkw) if top in T_ and _combo_ok(split, s_, positive=True)]
            fams_p = sorted({s_.family for s_, _ in pool})
            w = np.array([fw.get(f_, 0.0) for f_ in fams_p])
            if pool and w.sum() > 0:                                     # family re-drawn among those that can name the fruit
                fam_b = fams_p[int(rng.choice(len(fams_p), p=w / w.sum()))]
                singles = [(s_, T_) for s_, T_ in pool if s_.family == fam_b]
        s, T = singles[int(rng.randint(len(singles)))]
        return PairConfig(split.name, index, scene_seed, knobs, _episode(s, T, fruits, split.form, rng), None, None, s.family)
    raise RuntimeError(f"no valid config for {split.name}[{index}] (family {fam}) after {max_tries} tries")
