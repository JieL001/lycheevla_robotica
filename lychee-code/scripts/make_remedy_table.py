"""Remedy under bias and coverage control of the bias dial (exploratory arms).
   python scripts/make_remedy_table.py   ->  ../tables/remedy.tex, ../tables/remedy_macros.tex, results/eval/remedy.md
Rows: the unbiased arm R0(rho=0), the same data with its 'farthest' commands cut to the number that the dial leaves (146 of 8,000), the biased arm R0(rho=0.9),
the biased arm plus 2,000 extra PAIRED scenes and plus 4,000 extra unpaired NATURAL scenes (same number of samples), and R1.
Columns: PTA on IID, saliency-reversed and attribute-OOD scenes (mean over the available training seeds, range in brackets) and the accuracy on the depth commands that
name the 'farthest' and the 'nearest' fruit (IID and saliency-reversed scenes pooled, all seeds).  Macros: \\rp<Arm><Iid|Sal|Attr|Far|Near|FarN>."""
import json, os, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import per_seed, fmt

ARMS = [("Rzero", "r0_rho00_film", r"R0, $\rho=0$"),
        ("Far", "r0_rho00_far146", r"R0, $\rho=0$, 146 ``farthest''"),
        ("Matched", "r0_matched90_film", r"R0, command mix of $\rho=0.9$, no correlation"),
        ("Rninety", "r0_rho90_film", r"R0, $\rho=0.9$"),
        ("MixPaired", "mix_rho90_paired", r"R0, $\rho=0.9$ + paired scenes"),
        ("MixNatural", "mix_rho90_natural", r"R0, $\rho=0.9$ + relabelled scenes"),
        ("MixBiased", "mix_rho90_biased", r"R0, $\rho=0.9$ + biased scenes"),
        ("Rone", "r1_film", "R1")]
SPL = (("iid", "Iid"), ("sal", "Sal"), ("attr", "Attr"))


def depth_words(base):
    """Accuracy on 'farthest' and 'nearest' depth commands, IID and saliency-reversed scenes pooled over the available seeds: (n_far, acc_far, n_near, acc_near)."""
    nf = kf = nn = kn = 0
    for sp in ("iid", "sal"):
        for suf in ("", "_s1", "_s2", "_s3", "_s4"):
            p = f"results/eval/s_{base}{suf}__{sp}.jsonl"
            if not os.path.exists(p):
                continue
            for l in open(p):
                r = json.loads(l)
                if r["family"] != "mat_depth":
                    continue
                ok = r["first_detached"] >= 0 and r["first_detached"] in r["targets"]
                if "farthest" in r["text"]:
                    nf += 1; kf += ok
                elif "nearest" in r["text"]:
                    nn += 1; kn += ok
    return (nf, kf / nf if nf else float("nan"), nn, kn / nn if nn else float("nan"))


def main():
    rows, md, macros = [], ["| arm | seeds | IID | sal.-rev. | attr. | 'farthest' commands | acc. | 'nearest' commands | acc. |", "|---|---|---|---|---|---|---|---|---|"], []
    macros += [f"\\providecommand{{\\rp{k}{s}}}{{??}}" for k, _, _ in ARMS for s in ("Iid", "Sal", "Attr", "Far", "Near", "FarN")]
    for k, base, label in ARMS:
        cells, seeds = [], 0
        for sp, mname in SPL:
            ps = per_seed(base, sp)
            seeds = max(seeds, len(ps))
            if not ps:
                cells.append("--"); continue
            means = [float(np.mean(list(d.values()))) for d in ps]
            rng = f" ({fmt(min(means))}--{fmt(max(means))})" if len(ps) > 1 else ""
            cells.append(f"{fmt(np.mean(means))}{rng}")
            macros.append(f"\\renewcommand{{\\rp{k}{mname}}}{{{fmt(np.mean(means))}}}")
        if seeds == 0:
            continue
        if k == "Matched":
            macros.append(f"\\providecommand{{\\nsMatched}}{{{seeds}}}")               # training seeds of the mix-matched control
        nf, af, nn, an = depth_words(base)
        cells += [f"{100*af:.1f}", f"{100*an:.1f}"] if nf else ["--", "--"]
        if nf:
            macros.append(f"\\renewcommand{{\\rp{k}Far}}{{{100*af:.0f}}}\\renewcommand{{\\rp{k}Near}}{{{100*an:.0f}}}\\renewcommand{{\\rp{k}FarN}}{{{nf}}}")
        rows.append(f"{label} & {seeds} & " + " & ".join(cells))
        md.append(f"| {k} | {seeds} | " + " | ".join(cells[:3]) + f" | {nf} | {cells[3]} | {nn} | {cells[4]} |")
    macros.append("\\providecommand{\\rpMixBiasedDone}{" + ("1" if any(l.startswith("R0, $\\rho=0.9$ + biased scenes") for l in rows) else "0") + "}")
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/remedy.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/remedy_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/remedy.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
