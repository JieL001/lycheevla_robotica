"""PTA by command family of the selection-track arms, averaged over the available training seeds (mean and range over seeds), for one split.
   python scripts/make_family_seeds_table.py [--split attr]  ->  ../tables/family_<split>_seeds.tex, ../tables/family_seeds_macros.tex, results/eval/family_<split>_seeds.md
The seed-0 table (make_select_family_table.py) carries Wilson intervals over the scenes; this one shows how much a family value moves with the training seed, which is the
uncertainty that matters for held-out combinations (a single (family, maturity) cell of 65-83 pairs).  Macros: \\fs<Split><Arm><Family>[Lo|Hi] (percent)."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, ".")
from lychee import evalkit

FAMS = [("mat_any", "any-of", "Any"), ("mat_side", "side", "Side"), ("mat_ordinal", "ordinal", "Ord"), ("mat_depth", "depth", "Dep"), ("mat", "unique", "Uni")]
ARMS = [("Rone", "r1_film", "R1"), ("Ronenu", "r1u_film", "R1u"), ("Rsame", "r1n_film", "R1n"), ("RzeroP", "r0p_film", "R0p"),
        ("Rzero", "r0_rho00_film", r"R0, $\rho=0$"), ("Rninety", "r0_rho90_film", r"R0, $\rho=0.9$"), ("Far", "r0_rho00_far146", r"R0, $\rho=0$, 146 ``farthest''"),
        ("MixPaired", "mix_rho90_paired", r"R0, $\rho=0.9$ + paired"), ("MixNatural", "mix_rho90_natural", r"R0, $\rho=0.9$ + relabelled"),
        ("Rlate", "r1_late", "R1, late fusion"), ("Rtoken", "r1_token", "R1, command token"),
        ("RzeroLate", "r0_rho00_late", r"R0, $\rho=0$, late fusion"), ("RninetyLate", "r0_rho90_late", r"R0, $\rho=0.9$, late fusion"),
        ("Sone", "sym_r1", "R1, fruit table"), ("Szero", "sym_r0_rho00", r"R0, $\rho=0$, fruit table")]


def main(split):
    sp = split.capitalize()
    fams = [f for f in FAMS if not (split == "attr" and f[0] == "mat")]
    rows, md, macros = [], [f"| arm | seeds | " + " | ".join(n for _, n, _ in fams) + " |", "|---|---|" + "---|" * len(fams)], []
    for key, base, label in ARMS:
        per = []
        for suf in ("", "_s1", "_s2", "_s3", "_s4"):
            p = f"results/eval/s_{base}{suf}__{split}.jsonl"
            if os.path.exists(p):
                per.append(evalkit.summarize([json.loads(l) for l in open(p)], by="family"))
        if not per:
            continue
        cells = []
        for fk, fname, fmac in fams:
            v = [100 * s[fk]["PTA_sel"][0] for s in per if fk in s]
            if not v:
                cells.append("--"); continue
            rng = f" ({min(v):.0f}--{max(v):.0f})" if len(v) > 1 else ""
            cells.append(f"{np.mean(v):.0f}{rng}")
            macros.append(f"\\providecommand{{\\fs{sp}{key}{fmac}}}{{??}}\\renewcommand{{\\fs{sp}{key}{fmac}}}{{{np.mean(v):.0f}}}"
                          f"\\providecommand{{\\fs{sp}{key}{fmac}Lo}}{{??}}\\renewcommand{{\\fs{sp}{key}{fmac}Lo}}{{{min(v):.0f}}}"
                          f"\\providecommand{{\\fs{sp}{key}{fmac}Hi}}{{??}}\\renewcommand{{\\fs{sp}{key}{fmac}Hi}}{{{max(v):.0f}}}")
        rows.append(f"{label} & {len(per)} & " + " & ".join(cells))
        md.append(f"| {key} | {len(per)} | " + " | ".join(cells) + " |")
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    open(f"../tables/family_{split}_seeds.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open(f"../tables/family_{split}_seeds_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open(f"results/eval/family_{split}_seeds.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="attr")
    main(ap.parse_args().split)
