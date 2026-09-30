"""Measured properties of every rendered selection-track training set: the share of training commands whose target set contains the most salient fruit (salience = visibility /
depth^2 = visibility x projected radius^2, up to a constant), the number of distinct (scene, target set) labels, the share of scenes whose two commands have disjoint target sets (the
contrast), and the command mix (family shares; numbers of "farthest" and "nearest" depth commands).
   python scripts/bias_dial_measured.py   ->  results/bias_dial_measured.json and ../tables/dial_macros.tex (\\dial<Set>, \\labels<Set>, \\contrast<Set>, \\mix<Set><Family>, \\far<Set>, \\near<Set>)
The training sets are large and are not part of the repository: for a set that is not on disk the values of the shipped results/bias_dial_measured.json are used (so that the macro file
regenerates without the rendered data); for a set that is on disk they are measured again and the JSON is updated."""
import json, os, sys

import numpy as np

sys.path.insert(0, ".")
from lychee.paths import DATA

D = f"{DATA}/select"
JS = "results/bias_dial_measured.json"
SETS = [("rho00", "Rzero"), ("rho50", "Rhalf"), ("rho90", "Rninety"), ("rho97", "Rninetyseven"), ("rho99", "Rninetynine"),
        ("paired", "Rone"), ("indep", "Ronenu"), ("same", "Rsame"), ("matched90", "Rmatched")]
FAM_NAMES = ((0, "Unique"), (1, "Any"), (2, "Side"), (3, "Depth"), (4, "Ordinal"))


def measure(name):
    m = np.load(f"{D}/{name}/meta.npz", allow_pickle=True)
    ok = m["ok"].astype(bool)
    fvis, frad, fvalid = m["fvis"], m["frad"], m["fvalid"]
    tmask, text, family, ncmd = m["tmask"], m["text"], m["family"], m["ncmd"]        # load every array once (NpzFile decompresses on each access)
    top = (fvis * frad ** 2 * fvalid).argmax(1)
    hit, tot = 0, 0
    for k in range(2):
        rows = ok & (ncmd > k)
        hit += tmask[rows, k][np.arange(rows.sum()), top[rows]].sum()
        tot += rows.sum()
    r = dict(scenes=int(ok.sum()), commands=int(tot), share_top_in_targets=float(hit / tot))
    # distinct (scene, target set) labels: a paraphrase of the same command on the same scene adds none
    labels = set()
    for i in np.flatnonzero(ok):
        for k in range(int(ncmd[i])):
            labels.add((int(i), tuple(np.flatnonzero(tmask[i, k]))))
    r["distinct_labels"] = len(labels)
    texts = [str(text[i, k]) for i in np.flatnonzero(ok) for k in range(int(ncmd[i]))]
    fams = np.repeat(family[ok], ncmd[ok])
    for f, fname in FAM_NAMES:
        r[f"share_{fname}"] = float((fams == f).mean())
    r["farthest"], r["nearest"] = sum("farthest" in t for t in texts), sum("nearest" in t for t in texts)
    if (ncmd[ok] == 2).all():                                # two commands per scene: how many scenes carry a contrast (disjoint target sets)?
        a, b = tmask[ok, 0], tmask[ok, 1]
        r["share_disjoint_target_sets"] = float((~(a & b).any(1)).mean())
    return r


def main():
    res = json.load(open(JS)) if os.path.exists(JS) else {}
    for name, key in SETS:
        if not os.path.exists(f"{D}/{name}/meta.npz"):
            continue
        res[name] = r = measure(name)
        line = (f"{name:9s} scenes {r['scenes']:5d}  commands {r['commands']:5d}  labels {r['distinct_labels']:5d}  P(most salient fruit in target set) = {100*r['share_top_in_targets']:5.1f}%  "
                f"mix any {100*r['share_Any']:.0f}% depth {100*r['share_Depth']:.0f}%  farthest {r['farthest']} nearest {r['nearest']}")
        if "share_disjoint_target_sets" in r:
            line += f"  scenes with disjoint target sets = {100*r['share_disjoint_target_sets']:5.1f}%"
        print(line, flush=True)
    macros = []
    for name, key in SETS:
        if name not in res:
            continue
        r = res[name]
        macros.append(f"\\providecommand{{\\dial{key}}}{{??}}\\renewcommand{{\\dial{key}}}{{{100*r['share_top_in_targets']:.0f}}}")
        macros.append(f"\\providecommand{{\\labels{key}}}{{??}}\\renewcommand{{\\labels{key}}}{{{r['distinct_labels']}}}")
        for f, fname in FAM_NAMES:
            macros.append(f"\\providecommand{{\\mix{key}{fname}}}{{??}}\\renewcommand{{\\mix{key}{fname}}}{{{100*r['share_' + fname]:.0f}}}")
        macros.append(f"\\providecommand{{\\far{key}}}{{??}}\\renewcommand{{\\far{key}}}{{{r['farthest']}}}\\providecommand{{\\near{key}}}{{??}}\\renewcommand{{\\near{key}}}{{{r['nearest']}}}")
        if "share_disjoint_target_sets" in r:
            macros.append(f"\\providecommand{{\\contrast{key}}}{{??}}\\renewcommand{{\\contrast{key}}}{{{100*r['share_disjoint_target_sets']:.0f}}}")
    json.dump(res, open(JS, "w"), indent=1)
    os.makedirs("../tables", exist_ok=True)
    open("../tables/dial_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")


if __name__ == "__main__":
    main()
