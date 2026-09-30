"""Selection-track result tables from results/eval/s_<arm>__<split>.jsonl (+ probes s_<arm>_<mode>__iid.jsonl).
   python scripts/make_select_table.py   ->  ../tables/select_main.tex, results/eval/select_main.md

Columns: PTA on IID and on the saliency-reversed split (Wilson 95% intervals over scenes), TSA on IID, collapse rate on IID
(both commands lead to the same fruit), language sensitivity = TSA(normal) - TSA(swap) on IID (bootstrap over scenes), and
the TSA obtained with a blank command.
"""
import json, os, sys

import numpy as np

sys.path.insert(0, ".")
from lychee import evalkit

OUT = "../tables"
ARMS = [  # tag, label, data
    ("r0_rho00_film", r"R0, $\rho=0$", "unpaired, natural"),
    ("r0_rho50_film", r"R0, $\rho=0.5$", "unpaired, biased"),
    ("r0_rho90_film", r"R0, $\rho=0.9$", "unpaired, biased"),
    ("r0_rho97_film", r"R0, $\rho=0.97$", "unpaired, biased"),
    ("r0_rho99_film", r"R0, $\rho=0.99$", "unpaired, biased"),
    ("r1_film", "R1", "paired"),
    ("r1u_film", "R1u", "independent pair"),
    ("r1n_film", "R1n", "paraphrase pair"),
    ("r0p_film", "R0p", "first command of the pair"),
    ("r1_blank_film", "R1, blank command", "paired"),
    ("r1_late", r"R1, late fusion", "paired"),
    ("r1_token", r"R1, command token", "paired"),
    ("r0_rho00_late", r"R0, $\rho=0$, late fusion", "unpaired, natural"),
    ("r0_rho90_late", r"R0, $\rho=0.9$, late fusion", "unpaired, biased"),
    ("r0_rho00_token", r"R0, $\rho=0$, command token", "unpaired, natural"),
    ("r0_rho90_token", r"R0, $\rho=0.9$, command token", "unpaired, biased"),
    ("sym_r0_rho00", r"R0, $\rho=0$, fruit table", "unpaired, natural"),
    ("sym_r0_rho50", r"R0, $\rho=0.5$, fruit table", "unpaired, biased"),
    ("sym_r0_rho90", r"R0, $\rho=0.9$, fruit table", "unpaired, biased"),
    ("sym_r0_rho97", r"R0, $\rho=0.97$, fruit table", "unpaired, biased"),
    ("sym_r0_rho99", r"R0, $\rho=0.99$, fruit table", "unpaired, biased"),
    ("sym_r1", "R1, fruit table", "paired"),
    ("sym_r1u", "R1u, fruit table", "independent pair"),
    ("sym_r1_blank", "R1 blank, fruit table", "paired"),
]


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def ci(x, d=1):
    m, lo, hi = x
    return f"{100*m:.{d}f} [{100*lo:.0f}, {100*hi:.0f}]"


def sens(normal, swap):
    pn, ps = evalkit.pair_table(normal), evalkit.pair_table(swap)
    keys = sorted(set(pn) & set(ps))
    if not keys:
        return None
    tsa = lambda p: 0.5 * (float(p["plus"]["target_correct"]) + float(p["minus"]["target_correct"]))
    return evalkit.bootstrap_ci([tsa(pn[k]) - tsa(ps[k]) for k in keys])


def main():
    rows, md = [], []
    for tag, label, data in ARMS:
        iid, sal = load(f"results/eval/s_{tag}__iid.jsonl"), load(f"results/eval/s_{tag}__sal.jsonl")
        if not iid:
            continue
        si = evalkit.summarize(iid)["all"]
        ss = evalkit.summarize(sal)["all"] if sal else None
        swp, blk = load(f"results/eval/s_{tag}_swap__iid.jsonl"), load(f"results/eval/s_{tag}_blank__iid.jsonl")
        sn = sens(iid, swp) if swp else None
        if sn and "blank" in tag:                                          # a network trained without commands receives the empty command in every probe: sensitivity 0 by construction
            assert sn[0] == 0.0 and sn[1] == 0.0 and sn[2] == 0.0, f"{tag}: language sensitivity {sn} of a blank-command arm must be exactly 0 (see results/eval/blank_probe_audit.md)"
        sn_txt = f"{100*sn[0]:.0f} [{100*sn[1]:.0f}, {100*sn[2]:.0f}]" if sn else "--"
        blank_txt = f"{100*evalkit.summarize(blk)['all']['TSA'][0]:.0f}" if blk else "--"
        sal_txt = ci(ss["PTA_sel"]) if ss else "--"
        rows.append(f"{label} & {data} & {ci(si['PTA_sel'])} & {sal_txt} & {100*si['TSA'][0]:.1f} & {100*si['same_first_fruit'][0]:.0f} & {sn_txt} & {blank_txt}")
        md.append(f"| {tag} | {data} | {si['n_pairs']} | {ci(si['PTA_sel'])} | {sal_txt} | {100*si['TSA'][0]:.1f} | {100*si['same_first_fruit'][0]:.0f} | {sn_txt} | {blank_txt} |")
    if not rows:
        print("no selection-track results yet"); return
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/select_main.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")      # rows separated by \\ ; main.tex supplies the last one
    head = "| arm | data | pairs | PTA iid | PTA sal.-rev. | TSA iid | collapse | lang. sens. | TSA blank |\n|---|---|---|---|---|---|---|---|---|"
    text = head + "\n" + "\n".join(md)
    open("results/eval/select_main.md", "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text)
    chance = evalkit.summarize(load(f"results/eval/s_{ARMS[0][0]}__iid.jsonl"))["all"]["chance_PTA"]
    print(f"\nchance-level PTA on this sample: {100*chance:.1f}%")


if __name__ == "__main__":
    main()
