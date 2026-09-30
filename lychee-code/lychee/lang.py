"""Symbolic instruction layer for LycheeHarvest (simulator independent).

Targets are resolved from *privileged* per-fruit annotations; a policy never sees them.
An instruction is a ``Spec`` (family + attribute values).  ``resolve`` returns the unique target
fruit index, or ``None`` when the instruction is ambiguous / unpickable (which the generator must
never emit).  ``counterfactuals`` enumerates single-field edits that switch the target while
staying unique -- the paired-command contrast used for training data and for the diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

MATURITY = ("immature", "turning", "ripe")            # ids 0, 1, 2
FAMILIES = ("mat", "mat_side", "mat_depth", "mat_ordinal", "mat_any")
FIELDS = {                                            # discriminative fields per family
    "mat": ("mat",),
    "mat_side": ("mat", "side"),
    "mat_depth": ("mat", "depth"),
    "mat_ordinal": ("mat", "side", "k"),
    "mat_any": ("mat",),
}
CHOICES = {"mat": (0, 1, 2), "side": ("left", "right"), "depth": ("near", "far"), "k": (2, 3)}


@dataclass(frozen=True)
class FruitView:
    idx: int
    mat: int          # maturity id
    u: float          # image x in [0, 1], 0 = left edge
    depth: float      # metres from the camera
    vis: float        # visible fraction in [0, 1]


@dataclass(frozen=True)
class Spec:
    family: str
    mat: int
    side: Optional[str] = None     # "left" | "right"
    depth: Optional[str] = None    # "near" | "far"
    k: Optional[int] = None        # 1-based ordinal counted from `side`


def _extremal(cands: Sequence[FruitView], key, largest: bool, margin: float) -> Optional[int]:
    """Index of the extremal fruit if it beats the runner-up by ``margin``; else None."""
    if len(cands) < 2:
        return None
    order = sorted(cands, key=key, reverse=largest)
    if abs(key(order[0]) - key(order[1])) < margin:
        return None
    return order[0].idx


def resolve(fruits: Sequence[FruitView], spec: Spec, *, u_margin: float = 0.05,
            d_margin: float = 0.03, min_vis: float = 0.0, ord_min_vis: float = 0.0) -> Optional[int]:
    """Unique target index for ``spec`` or None (ambiguous, degenerate, or not pickable).
    ``ord_min_vis``: an ordinal command is valid only if every fruit of the commanded maturity up to position k (counted
    from the side) has at least this visible fraction, so that a viewer who counts the fruit they can see reaches the same
    k-th fruit as the label (0 = the counting includes fully hidden fruit)."""
    cands = [f for f in fruits if f.mat == spec.mat]
    if spec.family == "mat_any":
        return None                       # any-of instruction: use ``target_set``
    if spec.family == "mat":
        tgt = cands[0].idx if len(cands) == 1 else None
    elif spec.family == "mat_side":                   # "the ripe lychee on the left" = leftmost ripe
        tgt = _extremal(cands, lambda f: f.u, spec.side == "right", u_margin)
    elif spec.family == "mat_depth":
        tgt = _extremal(cands, lambda f: f.depth, spec.depth == "far", d_margin)
    elif spec.family == "mat_ordinal":
        if spec.k is None or len(cands) < spec.k:
            return None
        order = sorted(cands, key=lambda f: f.u, reverse=(spec.side == "right"))
        # ordering must be unambiguous up to position k (and against the (k+1)-th if present)
        upto = order[: spec.k + 1]
        if any(abs(a.u - b.u) < u_margin for a, b in zip(upto, upto[1:])):
            return None
        if any(f.vis < ord_min_vis for f in order[: spec.k]):
            return None
        tgt = order[spec.k - 1].idx
    else:
        raise ValueError(spec.family)
    if tgt is None:
        return None
    if next(f for f in fruits if f.idx == tgt).vis < min_vis:
        return None
    return tgt


def valid_specs(fruits: Sequence[FruitView], **kw) -> list[tuple[Spec, int]]:
    """Every instruction (over all families / attribute values) that resolves uniquely."""
    out = []
    for m in CHOICES["mat"]:
        out.append(Spec("mat", m))
        for s in CHOICES["side"]:
            out.append(Spec("mat_side", m, side=s))
            for k in CHOICES["k"]:
                out.append(Spec("mat_ordinal", m, side=s, k=k))
        for d in CHOICES["depth"]:
            out.append(Spec("mat_depth", m, depth=d))
    res = [(s, resolve(fruits, s, **kw)) for s in out]
    return [(s, t) for s, t in res if t is not None]


def counterfactuals(fruits: Sequence[FruitView], spec: Spec, **kw) -> list[tuple[Spec, int, str]]:
    """Single-field edits of ``spec`` that switch the target and stay unique: (spec', target', field)."""
    t = resolve(fruits, spec, **kw)
    if t is None:
        return []
    out = []
    for field in FIELDS[spec.family]:
        for c in CHOICES[field]:
            if c == getattr(spec, field):
                continue
            s2 = replace(spec, **{field: c})
            t2 = resolve(fruits, s2, **kw)
            if t2 is not None and t2 != t:
                out.append((s2, t2, field))
    return out


def counterfactual_pairs(fruits: Sequence[FruitView], **kw) -> list[tuple[Spec, int, Spec, int, str]]:
    """All (spec+, target+, spec-, target-, edited_field) available in one scene."""
    pairs = []
    for s, t in valid_specs(fruits, **kw):
        for s2, t2, f in counterfactuals(fruits, s, **kw):
            pairs.append((s, t, s2, t2, f))
    return pairs


# ----------------------------------------------------------------------------------------------
# Target *sets*.  Unique families have a one-element set; the any-of family ``mat_any`` ("pick any ripe
# lychee") accepts every visible fruit of the commanded maturity -- provided a different maturity is
# also present, otherwise the command would be trivial.  Two readings of "visible" (``any_rule``):
#   "v1.0"  a fruit of the maturity is a target if its visibility lies in [min_vis, max_vis], the target range of the split
#           (all results of the paper).  On the occlusion split (0.2-0.4) a well-visible fruit of the named maturity is then
#           NOT a target, which the command does not say;
#   "v1.1"  every fruit of the maturity with visibility >= min_vis is a target, and the command is issued only if no fruit of
#           the maturity is more visible than max_vis: occlusion is a property of the scene, not of the legal answers.
#           On splits with max_vis = 1 the two rules coincide.
# ----------------------------------------------------------------------------------------------
ANY_RULES = ("v1.0", "v1.1")


def target_set(fruits: Sequence[FruitView], spec: Spec, *, u_margin: float = 0.05, d_margin: float = 0.03,
               min_vis: float = 0.0, max_vis: float = 1.0, ord_min_vis: float = 0.0, any_rule: str = "v1.0") -> tuple:
    if any_rule not in ANY_RULES:
        raise ValueError(f"any_rule must be one of {ANY_RULES}, not {any_rule!r}")
    if spec.family == "mat_any":
        named = [f for f in fruits if f.mat == spec.mat]
        if any_rule == "v1.1":
            if any(f.vis > max_vis for f in named):
                return ()
            cands = tuple(f.idx for f in named if f.vis >= min_vis)
        else:
            cands = tuple(f.idx for f in named if min_vis <= f.vis <= max_vis)
        return cands if cands and any(f.mat != spec.mat for f in fruits) else ()
    t = resolve(fruits, spec, u_margin=u_margin, d_margin=d_margin, min_vis=min_vis, ord_min_vis=ord_min_vis)
    if t is None or next(f for f in fruits if f.idx == t).vis > max_vis:
        return ()
    return (t,)


def all_specs() -> list:
    out = []
    for m in CHOICES["mat"]:
        out.append(Spec("mat", m))
        out.append(Spec("mat_any", m))
        for s in CHOICES["side"]:
            out.append(Spec("mat_side", m, side=s))
            for k in CHOICES["k"]:
                out.append(Spec("mat_ordinal", m, side=s, k=k))
        for d in CHOICES["depth"]:
            out.append(Spec("mat_depth", m, depth=d))
    return out


def valid_specs_sets(fruits: Sequence[FruitView], **kw) -> list:
    """[(spec, targets)] for every instruction with a non-empty target set."""
    res = [(s, target_set(fruits, s, **kw)) for s in all_specs()]
    return [(s, T) for s, T in res if T]


def counterfactual_pairs_sets(fruits: Sequence[FruitView], **kw) -> list:
    """[(spec+, T+, spec-, T-, edited_field)]: one field edited, both target sets non-empty and disjoint."""
    out = []
    for s, T in valid_specs_sets(fruits, **kw):
        for field in FIELDS[s.family]:
            for c in CHOICES[field]:
                if c == getattr(s, field):
                    continue
                s2 = replace(s, **{field: c})
                T2 = target_set(fruits, s2, **kw)
                if T2 and not (set(T) & set(T2)):
                    out.append((s, T, s2, T2, field))
    return out


# ----------------------------------------------------------------------------------------------
# Surface forms.  Set "A" is used for training, set "B" replaces the sentence frames, the depth phrases and most maturity
# words (language-OOD split).  The shift is partial: the tokens "green" and "red" (in "young green", "fully red"), the side
# words and the ordinal words occur in both sets.
# ----------------------------------------------------------------------------------------------
MAT_WORDS = {
    "A": {0: ("unripe", "green"), 1: ("turning", "half-ripe"), 2: ("ripe", "red")},
    "B": {0: ("immature", "young green"), 1: ("yellowing", "ripening"), 2: ("mature", "fully red")},
}
SIDE_WORDS = {"A": {"left": "left", "right": "right"},
              "B": {"left": "left-hand side", "right": "right-hand side"}}
DEPTH_WORDS = {"A": {"near": "nearest", "far": "farthest"},
               "B": {"near": "closest to the camera", "far": "furthest from the camera"}}
ORD_WORDS = {2: "second", 3: "third"}
FRAMES = {
    "A": {
        "mat": ("pick the {m} lychee", "harvest the {m} lychee", "grab the {m} lychee"),
        "mat_side": ("pick the {m} lychee on the {s}", "harvest the {m} lychee on the {s}"),
        "mat_depth": ("pick the {d} {m} lychee", "harvest the {d} {m} lychee"),
        "mat_ordinal": ("pick the {o} {m} lychee from the {s}", "harvest the {o} {m} lychee from the {s}"),
        "mat_any": ("pick any {m} lychee", "harvest any {m} lychee"),
    },
    "B": {
        "mat": ("please take the {m} lychee", "go for the {m} lychee"),
        "mat_side": ("please take the {m} lychee that is on the {s}", "go for the {m} lychee toward the {s}"),
        "mat_depth": ("please take the {m} lychee that is {d}", "go for the {m} lychee that is {d}"),
        "mat_ordinal": ("please take the {o} {m} lychee counting from the {s}",),
        "mat_any": ("please take any {m} lychee", "go for any {m} lychee"),
    },
}


def render(spec: Spec, *, form: str = "A", variant: int = 0, word: int = 0,
           avoid: Optional[int] = None) -> str:
    """English command for ``spec``.  ``avoid`` appends a collision constraint on that maturity."""
    m = MAT_WORDS[form][spec.mat][word % 2]
    frames = FRAMES[form][spec.family]
    text = frames[variant % len(frames)].format(
        m=m,
        s=SIDE_WORDS[form].get(spec.side, ""),
        d=DEPTH_WORDS[form].get(spec.depth, ""),
        o=ORD_WORDS.get(spec.k, ""),
    )
    if avoid is not None:
        text += f" without touching the {MAT_WORDS[form][avoid][0]} fruits"
    return text
