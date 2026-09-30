"""Attribute-OOD split: accuracy on the commands that ARE held-out (family, maturity) combinations, and PTA on the pairs in which both commands are held-out combinations.
   python scripts/attr_heldout_only.py
In the attribute-OOD split only the FIRST command of a pair (`plus`) is required to be a held-out combination; the second is a seen combination in the pairs that edit the
maturity (54-57 % of the pairs).  This script therefore reports, per arm (mean over the training seeds, range in brackets), (i) the accuracy TSA on the first commands only
(unseen combinations only) and (ii) PTA on the pairs that edit the side, depth or ordinal (both commands unseen), for the pilot (attr) and fresh (attrf) scenes.
Writes ../tables/attr_heldout.tex (rows), ../tables/attr_heldout_macros.tex (\\ho<Arm><Set>{Tsa,TsaLo,TsaHi,Both,BothN}) and results/eval/attr_heldout.md."""
import json, os, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import ARMS, SEED_SUFFIXES

SHOW = ["Rone", "Ronenu", "Rsame", "RzeroP", "Rzero", "Rninety", "Rlate", "Rtoken", "RoneLateWide", "Sone", "Szero"]
SETS = [("attr", "pilot", "Pil"), ("attrf", "fresh", "Fresh")]
LABEL = {k: l for k, _, l in ARMS}
LABEL.update({"RoneLateWide": r"R1, late fusion (0.60 M)"})
BASE = {k: b for k, b, _ in ARMS}
BASE.update({"RoneLateWide": "r1_late_wide"})


def load_seeds(base, split):
    out = []
    for suf in SEED_SUFFIXES:
        p = f"results/eval/s_{base}{suf}__{split}.jsonl"
        if os.path.exists(p):
            out.append([json.loads(l) for l in open(p)])
    return out


def fmt_range(vals):
    m = 100 * np.mean(vals)
    return f"{m:.1f}" + (f" ({100*min(vals):.1f}--{100*max(vals):.1f})" if len(vals) > 1 else "")


def main():
    rows, macros, md = [], [], ["| arm | seeds | first commands (unseen combination): accuracy pilot | fresh | pairs with both commands unseen: PTA pilot | fresh |", "|---|---|---|---|---|---|"]
    macros += [f"\\providecommand{{\\ho{key}{tag}{q}}}{{??}}" for key in SHOW for _, _, tag in SETS for q in ("Tsa", "TsaLo", "TsaHi", "Both", "BothN")]      # shown as ?? until an arm has records
    for key in SHOW:
        cells, mdcells, n_seeds = [], [], 0
        for split, name, tag in SETS:
            seeds = load_seeds(BASE[key], split)
            if not seeds:
                cells += ["--", "--"]; mdcells += ["--", "--"]
                continue
            n_seeds = max(n_seeds, len(seeds))
            tsa, both = [], []
            for recs in seeds:
                first = [r for r in recs if r["which"] == "plus"]
                tsa.append(np.mean([r["target_correct"] for r in first]))
                pairs = {}
                for r in recs:
                    pairs.setdefault(r["index"], {})[r["which"]] = r
                sel = [v for v in pairs.values() if v["plus"]["field"] != "mat" and "minus" in v]
                both.append(np.mean([all(v[w]["target_correct"] for w in ("plus", "minus")) for v in sel]))
                n_both = len(sel)
            cells += [fmt_range(tsa), fmt_range(both)]
            mdcells += [fmt_range(tsa), fmt_range(both)]
            macros.append(f"\\renewcommand{{\\ho{key}{tag}Tsa}}{{{100*np.mean(tsa):.1f}}}\\renewcommand{{\\ho{key}{tag}TsaLo}}{{{100*min(tsa):.1f}}}\\renewcommand{{\\ho{key}{tag}TsaHi}}{{{100*max(tsa):.1f}}}"
                          f"\\renewcommand{{\\ho{key}{tag}Both}}{{{100*np.mean(both):.1f}}}\\renewcommand{{\\ho{key}{tag}BothN}}{{{n_both}}}")
        if n_seeds == 0:
            continue
        # column order: pilot TSA, pilot both, fresh TSA, fresh both
        rows.append(f"{LABEL[key]} & {n_seeds} & {cells[0]} & {cells[2]} & {cells[1]} & {cells[3]}")
        md.append(f"| {key} | {n_seeds} | {mdcells[0]} | {mdcells[2]} | {mdcells[1]} | {mdcells[3]} |")
    os.makedirs("../tables", exist_ok=True)
    open("../tables/attr_heldout.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/attr_heldout_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/attr_heldout.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
