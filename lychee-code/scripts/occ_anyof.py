"""Any-of pairs on the occlusion split under the two readings of the command -- re-scoring of the recorded selections (no retraining, no simulation, no rendering).
   python scripts/occ_anyof.py   ->  ../tables/occ_anyof.tex, ../tables/occ_anyof_macros.tex, results/eval/occ_anyof.md
Benchmark v1.0 (all results of the paper) defines the targets of an any-of command as the fruit of the named maturity whose visibility lies in the target range of the split, 0.2-0.4 on the
occlusion split, so a well-visible fruit of the named maturity is not a target although "any" says nothing about visibility.  Benchmark v1.1 (lychee.lang.target_set, any_rule="v1.1") accepts
every fruit of the maturity that is visible enough (>= 0.2) and issues the command on the occlusion split only if no fruit of that maturity is more visible than 0.4, so that occlusion is a
property of the scene, not of the legal answers.  This script scores the recorded selections (first_detached) of every arm three ways:
  as scored      the v1.0 target sets (the stored ones; recomputed from the shipped fruit tables and required to be identical),
  revised rule   every fruit of the named maturity with visibility >= 0.2 is a target (what the words of the command mean),
  unambiguous    only the pairs that v1.1 would issue on such a scene (no fruit of either named maturity above 0.4); on them the two rules give the same target sets.
A random choice among the fruit of the scene is shown for reference (expected PTA of two independent draws)."""
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from lychee import lang
from lychee.modular import parse_command
from lychee.paths import eval_meta
from lychee.splits import SPLITS
from make_matched_dataset import views_of
from select_seeds import ARMS, SEED_SUFFIXES

LO, HI = SPLITS["occ_ood"].target_vis
ORDER = ["Rone", "Rzero", "Rninety", "Rlate", "Rtoken", "Sone", "ModFree", "ModDet", "ModOracle"]


def any_of_pairs():
    """{scene index: dict(v10=[T+, T-], rev=[T+, T-], unamb=bool, n=number of fruit)} of the any-of pairs of occ_300; the stored target masks must equal the recomputed v1.0 sets."""
    d = np.load(eval_meta("occ_300"))
    out = {}
    for i in np.flatnonzero(d["ok"]):
        specs = [parse_command(str(d["text"][i, k])) for k in range(int(d["ncmd"][i]))]
        if specs[0].family != "mat_any":
            continue
        views = views_of(d["fxy"][i], d["frad"][i], d["fmat"][i], d["fvis"][i], d["fvalid"][i])
        v10 = [tuple(lang.target_set(views, s, min_vis=LO, max_vis=HI)) for s in specs]
        stored = [tuple(int(j) for j in np.flatnonzero(d["tmask"][i, k])) for k in range(len(specs))]
        assert v10 == stored, f"scene {int(d['index'][i])}: stored targets {stored} differ from the recomputed v1.0 sets {v10}"
        rev = [tuple(lang.target_set(views, s, min_vis=LO, max_vis=1.0)) for s in specs]
        unamb = all(lang.target_set(views, s, min_vis=LO, max_vis=HI, any_rule="v1.1") for s in specs)
        if unamb:
            assert v10 == rev, "on an unambiguous pair the two rules must give the same target sets"
        out[int(d["index"][i])] = dict(v10=v10, rev=rev, unamb=unamb, n=len(views))
    return out


def score(pairs, recs):
    """(PTA as scored, PTA revised, PTA unambiguous) of one record file over the any-of pairs, and the pair counts."""
    by = {}
    for r in recs:
        if r["index"] in pairs:
            by.setdefault(r["index"], {})[r["which"]] = r
    a = b = c = nu = n = 0
    for i, v in by.items():
        if len(v) < 2:
            continue
        p, m = pairs[i], (v["plus"], v["minus"])
        sel = [x["first_detached"] for x in m]
        old = all(s in T for s, T in zip(sel, p["v10"]))
        assert old == all(x["target_correct"] for x in m), f"scene {i}: the recorded correctness differs from the stored v1.0 target sets"
        new = all(s in T for s, T in zip(sel, p["rev"]))
        n += 1
        a += old
        b += new
        if p["unamb"]:
            nu += 1
            c += old
            assert old == new
    return 100 * a / n, 100 * b / n, (100 * c / nu if nu else float("nan")), n, nu


def main():
    import json
    pairs = any_of_pairs()
    n, nu = len(pairs), sum(p["unamb"] for p in pairs.values())
    md = ["| arm | seeds | as scored (v1.0) | revised rule | unambiguous pairs |", "|---|---|---|---|---|"]
    rows, macros = [], [f"\\providecommand{{\\occAnyPairsN}}{{{n}}}", f"\\providecommand{{\\occAnyUnambN}}{{{nu}}}", f"\\providecommand{{\\occAnyAmbN}}{{{n - nu}}}",
                        f"\\providecommand{{\\occAnyAmbShare}}{{{100 * (n - nu) / n:.0f}}}", f"\\providecommand{{\\occAnyUnambShare}}{{{100 * nu / n:.0f}}}"]
    # random choice among the fruit of the scene
    ch = [np.mean([np.prod([len(T) / p["n"] for T in p[k]]) for p in pairs.values() if sel_ok(p, u)]) * 100 for k, u in (("v10", False), ("rev", False), ("v10", True))]
    rows.append(r"Random fruit & -- & " + " & ".join(f"{x:.1f}" for x in ch))
    md.append("| random | -- | " + " | ".join(f"{x:.1f}" for x in ch) + " |")
    macros += [f"\\providecommand{{\\oaChance{k}}}{{{x:.1f}}}" for k, x in zip(("Old", "New", "Unamb"), ch)]
    labels = {key: (base, label) for key, base, label in ARMS}
    for key in ORDER:
        base, label = labels[key]
        vals = []
        for suf in SEED_SUFFIXES:
            p = f"results/eval/s_{base}{suf}__occ.jsonl"
            if os.path.exists(p):
                vals.append(score(pairs, [json.loads(l) for l in open(p)]))
        if not vals:
            continue
        m = np.mean([v[:3] for v in vals], axis=0)
        cells = [f"{x:.1f}" for x in m]
        rows.append(f"{label} & {len(vals)} & " + " & ".join(cells))
        md.append(f"| {key} | {len(vals)} | " + " | ".join(cells) + " |")
        macros += [f"\\providecommand{{\\oa{key}{k}}}{{{x:.1f}}}" for k, x in zip(("Old", "New", "Unamb"), m)]
    os.makedirs("../tables", exist_ok=True)
    open("../tables/occ_anyof.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/occ_anyof_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    head = f"any-of pairs on occ_300: {n}; unambiguous (v1.1 would issue them): {nu} ({100 * nu / n:.0f} %); ambiguous: {n - nu} ({100 * (n - nu) / n:.0f} %)"
    open("results/eval/occ_anyof.md", "w", encoding="utf-8", newline="\n").write(head + "\n\n" + "\n".join(md) + "\n")
    print(head + "\n" + "\n".join(md))


def sel_ok(p, unamb_only):
    return p["unamb"] or not unamb_only


if __name__ == "__main__":
    main()
