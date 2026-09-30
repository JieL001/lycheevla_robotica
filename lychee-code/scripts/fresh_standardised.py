"""Pilot against fresh IID scenes: how much of the drop of PTA is a change of the command-family mix and how much is a change within the families?
   python scripts/fresh_standardised.py [--boot 4000]
For every pixel-selector arm (mean over its training seeds per pair) PTA on the fresh IID scenes is re-weighted to the family mix of the pilot IID scenes (direct standardisation):
    total change   = PTA(fresh) - PTA(pilot)
    mix effect     = PTA(fresh) - PTA(fresh, pilot mix)        (a different mix of families in the fresh sample)
    within effect  = PTA(fresh, pilot mix) - PTA(pilot)        (the same families are answered worse on the fresh scenes)
The two scene samples are independent; the 95 % intervals of R1 come from a bootstrap over scenes of both samples.
Writes ../tables/fresh_std.tex (rows), ../tables/fresh_std_macros.tex (\\fs<Arm>{Tot,Mix,Within}, \\fsMinWithin, \\fsMaxWithin over arms, \\fsRone{Within}{Lo,Hi}) and results/eval/fresh_standardised.md."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import ARMS, SEED_SUFFIXES

FAMS = ["mat", "mat_any", "mat_side", "mat_depth", "mat_ordinal"]
SKIP = ("Sone", "Szero", "Sninety", "Sninetyseven", "Sninetynine", "MixBiased", "Far", "MixPaired", "MixNatural", "RzeroLate", "RninetyLate", "RzeroToken", "RninetyToken", "ModOracle", "ModDet", "ModFree",
        "RoneNoShift", "RoneSyncShift", "RlateNoShift", "RlateSyncShift", "RlateWide", "RoneLrLo", "RoneLrHi", "RlateLrLo", "RlateLrHi")   # the modular baseline has no training seeds of a selector; the recipe controls of Table 11 are not part of this comparison


def per_pair(base, split):
    """{index: (family, mean over seeds of the pair outcome)} and the number of seeds."""
    acc, fam, n = {}, {}, 0
    for suf in SEED_SUFFIXES:
        p = f"results/eval/s_{base}{suf}__{split}.jsonl"
        if not os.path.exists(p):
            continue
        n += 1
        pairs = {}
        for r in map(json.loads, open(p)):
            pairs.setdefault(r["index"], {})[r["which"]] = r
        for i, v in pairs.items():
            if "plus" in v and "minus" in v:
                acc.setdefault(i, []).append(all(v[w]["target_correct"] for w in ("plus", "minus")))
                fam[i] = v["plus"]["family"]
    return {i: (fam[i], float(np.mean(x))) for i, x in acc.items()}, n


def by_family(d):
    return {f: np.array([y for (ff, y) in d.values() if ff == f]) for f in FAMS}


def standardised(fresh_f, w):
    return float(sum(w[f] * fresh_f[f].mean() for f in FAMS if len(fresh_f[f])))


def main(n_boot):
    rng = np.random.RandomState(0)
    rows, md, macros = [], ["| arm | seeds | pilot | fresh | fresh with pilot family mix | total | mix effect | within-family effect |", "|---|---|---|---|---|---|---|---|"], []
    within_all, mix_all, tot_all = [], [], []
    for key, base, label in ARMS:
        if key in SKIP:
            continue
        P, ns_p = per_pair(base, "iid")
        Fh, ns_f = per_pair(base, "iidf")
        if not P or not Fh:
            continue
        pf, ff = by_family(P), by_family(Fh)
        w = {f: len(pf[f]) / sum(len(pf[g]) for g in FAMS) for f in FAMS}
        pil = sum(w[f] * pf[f].mean() for f in FAMS)                                  # = the pilot PTA (weights are its own mix)
        fresh = float(np.mean([y for (_, y) in Fh.values()]))
        std = standardised(ff, w)
        tot, mix, within = fresh - pil, fresh - std, std - pil
        within_all.append((within, key)); mix_all.append(mix); tot_all.append(tot)
        lo = hi = None
        if key == "Rone":
            ws = []
            for _ in range(n_boot):
                bp = {f: pf[f][rng.randint(0, len(pf[f]), len(pf[f]))].mean() for f in FAMS}
                bf = {f: ff[f][rng.randint(0, len(ff[f]), len(ff[f]))].mean() for f in FAMS}
                ws.append(sum(w[f] * bf[f] for f in FAMS) - sum(w[f] * bp[f] for f in FAMS))
            lo, hi = np.percentile(ws, [2.5, 97.5])
            macros.append(f"\\providecommand{{\\fsRoneWithinLo}}{{{100*lo:+.1f}}}\\providecommand{{\\fsRoneWithinHi}}{{{100*hi:+.1f}}}".replace("-", "$-$"))
        f1 = lambda v: f"{100*v:+.1f}".replace("-", "$-$")
        rows.append(f"{label} & {ns_p}/{ns_f} & {100*pil:.1f} & {100*fresh:.1f} & {100*std:.1f} & {f1(tot)} & {f1(mix)} & {f1(within)}" + (f" [{f1(lo)}, {f1(hi)}]" if lo is not None else ""))
        md.append(f"| {key} | {ns_p}/{ns_f} | {100*pil:.1f} | {100*fresh:.1f} | {100*std:.1f} | {f1(tot)} | {f1(mix)} | {f1(within)}" + (f" [{f1(lo)}, {f1(hi)}]" if lo is not None else "") + " |")
        macros.append(f"\\providecommand{{\\fs{key}Tot}}{{{f1(tot)}}}\\providecommand{{\\fs{key}Mix}}{{{f1(mix)}}}\\providecommand{{\\fs{key}Within}}{{{f1(within)}}}")
    ws = [v for v, _ in within_all]
    macros.append(f"\\providecommand{{\\fsMinWithin}}{{{100*min(ws):+.1f}}}\\providecommand{{\\fsMaxWithin}}{{{100*max(ws):+.1f}}}\\providecommand{{\\fsNArms}}{{{len(ws)}}}".replace("-", "$-$"))
    # sizes of the drops (all negative here) as plain numbers, for sentences of the form "removes x to y points of the z to w points of the drop"
    ab = lambda v: [100 * abs(x) for x in v]
    macros.append(f"\\providecommand{{\\fsTotAbsMin}}{{{min(ab(tot_all)):.1f}}}\\providecommand{{\\fsTotAbsMax}}{{{max(ab(tot_all)):.1f}}}"
                  f"\\providecommand{{\\fsMixAbsMin}}{{{min(ab(mix_all)):.1f}}}\\providecommand{{\\fsMixAbsMax}}{{{max(ab(mix_all)):.1f}}}"
                  f"\\providecommand{{\\fsWithinAbsMin}}{{{min(ab(ws)):.1f}}}\\providecommand{{\\fsWithinAbsMax}}{{{max(ab(ws)):.1f}}}")
    os.makedirs("../tables", exist_ok=True)
    open("../tables/fresh_std.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/fresh_std_macros.tex", "w", newline="\n").write("\n".join(macros).replace("+-", "$-$") + "\n")
    open("results/eval/fresh_standardised.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    main(ap.parse_args().boot)
