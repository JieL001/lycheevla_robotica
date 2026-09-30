"""Sensitivity of the selection-track results to the snap radius of the pixel-to-fruit rule (20 px in all reported results).
   python scripts/snap_summary.py
Reads s_<arm>[_sK]_snap<R>__<split>.jsonl (R = 10, 30, 40; lane_snap.sh) and the reported records (R = 20).  For every arm group (mean over the training seeds) it lists PTA on the IID,
saliency-reversed and attribute-OOD scenes at each radius, the rank correlation of the arms across radii, and the sign of the comparisons of the paper (C1 dial cost, pairing effect,
FiLM against late fusion and the command token).  Writes ../tables/snap.tex (rows), ../tables/snap_macros.tex and results/eval/snap.md."""
import json, os, sys

import numpy as np
from scipy import stats as sst

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import SEED_SUFFIXES

ARMS = [("Rzero", "r0_rho00_film", r"R0, $\rho=0$"), ("Rninety", "r0_rho90_film", r"R0, $\rho=0.9$"), ("Rninetyseven", "r0_rho97_film", r"R0, $\rho=0.97$"),
        ("Rninetynine", "r0_rho99_film", r"R0, $\rho=0.99$"), ("Rone", "r1_film", "R1"), ("Ronenu", "r1u_film", "R1u"), ("Rlate", "r1_late", "R1, late fusion"),
        ("Rtoken", "r1_token", "R1, command token")]
RADII = [10, 20, 30, 40]
RNAME = {10: "Ten", 20: "Twenty", 30: "Thirty", 40: "Forty"}                # macro names cannot contain digits
CMP = [("C1 dial cost (sal): R0(0.9) - R0(0)", "Rninety", "Rzero", "sal", "SnapCone"), ("pairing: R1 - R0(0), IID", "Rone", "Rzero", "iid", "SnapPair"),
       ("C3: R1 - R1u, IID", "Rone", "Ronenu", "iid", "SnapCthree"), ("FiLM - late fusion, attr", "Rone", "Rlate", "attr", "SnapLate"), ("FiLM - command token, attr", "Rone", "Rtoken", "attr", "SnapTok")]
SPL = [("iid", "Iid"), ("sal", "Sal"), ("attr", "Attr")]


def pta_seeds(base, split, radius):
    suffix = "" if radius == 20 else f"_snap{radius}"
    out = []
    for suf in SEED_SUFFIXES:
        p = f"results/eval/s_{base}{suf}{suffix}__{split}.jsonl"
        if not os.path.exists(p):
            continue
        pairs = {}
        for r in map(json.loads, open(p)):
            pairs.setdefault(r["index"], {})[r["which"]] = r
        out.append(float(np.mean([all(v[w]["target_correct"] for w in ("plus", "minus")) for v in pairs.values() if "minus" in v])))
    return out


def main():
    table, rows, macros, md = {}, [], [], ["| arm | seeds | split | r=10 | r=20 | r=30 | r=40 |", "|---|---|---|---|---|---|---|"]
    macros += [f"\\providecommand{{\\sn{key}{tag}R{RNAME[R]}}}{{??}}" for key, _, _ in ARMS for _, tag in SPL for R in RADII]                       # ?? until the records exist
    macros += [f"\\providecommand{{\\snRho{tag}Min}}{{??}}" for _, tag in SPL] + [f"\\providecommand{{\\sn{m}{e}}}{{??}}" for *_, m in CMP for e in ("Lo", "Hi")]
    for key, base, label in ARMS:
        cells = {}
        for sp, tag in SPL:
            vals = {R: pta_seeds(base, sp, R) for R in RADII}
            if not all(vals.values()):
                continue
            n = min(len(v) for v in vals.values())
            cells[sp] = {R: 100 * float(np.mean(v[:n])) for R, v in vals.items()}
            table[(key, sp)] = cells[sp]
            md.append(f"| {key} | {n} | {sp} | " + " | ".join(f"{cells[sp][R]:.1f}" for R in RADII) + " |")
            for R in RADII:
                macros.append(f"\\renewcommand{{\\sn{key}{tag}R{RNAME[R]}}}{{{cells[sp][R]:.1f}}}")
        if len(cells) == len(SPL):
            rows.append(f"{label} & " + " & ".join(" / ".join(f"{cells[sp][R]:.1f}" for R in RADII) for sp, _ in SPL))
    # largest change of the mean PTA of any arm on any of the splits between the radius of the paper (20 px) and the other radii
    macros += [f"\\providecommand{{\\snMax{RNAME[R]}}}{{??}}" for R in RADII if R != 20]
    for R in RADII:
        if R == 20 or not table:
            continue
        worst = max(abs(c[R] - c[20]) for c in table.values())
        macros.append(f"\\renewcommand{{\\snMax{RNAME[R]}}}{{{worst:.1f}}}")
        md.append(f"\nlargest change of the mean PTA of an arm on a split between r=20 and r={R}: {worst:.1f} points")
    # rank correlation of the arms across radii, per split
    for sp, tag in SPL:
        keys = [k for k, _, _ in ARMS if (k, sp) in table]
        if len(keys) < 4:
            continue
        base = [table[(k, sp)][20] for k in keys]
        rhos = [sst.spearmanr(base, [table[(k, sp)][R] for k in keys])[0] for R in RADII if R != 20]
        macros.append(f"\\renewcommand{{\\snRho{tag}Min}}{{{min(rhos):.2f}}}")
        md.append(f"\nSpearman rank correlation of the {len(keys)} arms between r=20 and r=10/30/40 on {sp}: " + ", ".join(f"{r:.2f}" for r in rhos))
    # the sign of the comparisons of the paper at every radius
    cmp_ = CMP
    md.append("\n| comparison | r=10 | r=20 | r=30 | r=40 |\n|---|---|---|---|---|")
    for name, a, b, sp, m in cmp_:
        if (a, sp) in table and (b, sp) in table:
            d = {R: table[(a, sp)][R] - table[(b, sp)][R] for R in RADII}
            md.append(f"| {name} | " + " | ".join(f"{d[R]:+.1f}" for R in RADII) + " |")
            macros.append(f"\\renewcommand{{\\sn{m}Lo}}{{{min(d.values()):+.1f}}}\\renewcommand{{\\sn{m}Hi}}{{{max(d.values()):+.1f}}}".replace("+-", "$-$").replace("{-", "{$-$"))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/snap.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/snap_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/snap.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
