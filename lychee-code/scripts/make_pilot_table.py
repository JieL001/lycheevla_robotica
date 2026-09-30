"""Compact-policy pilot table from results/eval/bc_<arm>__<split>.jsonl (+ the swap probe bc_<arm>_swap__<split>.jsonl).
   python scripts/make_pilot_table.py [--split iid]   ->  ../tables/pilot_<split>.tex  and  results/eval/pilot_<split>.md

Columns: PTA (first detached fruit is a target, both commands), per-command target accuracy, wrong-target rate,
harvest success, PTA-approach (diagnostic), and language sensitivity = TSA(normal) - TSA(swap) on the scenes that
were evaluated under both (swap probe feeds the OTHER member's command of the pair; a language-blind policy scores 0).
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, ".")
from lychee import evalkit

OUT = "../tables"

ARMS = [  # tag, data, conditioning
    ("r0_late", "unpaired", "late fusion"),
    ("r0_film", "unpaired", "FiLM"),
    ("r1_late", "paired", "late fusion"),
    ("r1_film", "paired", "FiLM"),
    ("r5_full", "paired", "grounding: supervised + injected"),
    ("r5_sup_only", "paired", "grounding: supervision only"),
    ("r5_inj_only", "paired", "grounding: injection only"),
    ("r5_full_lang", "paired", "grounding + direct language path"),
]


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def pct(ci, d=1):
    m, lo, hi = ci
    return f"{100*m:.{d}f} [{100*lo:.0f}, {100*hi:.0f}]"


def sensitivity(normal, swap):
    """TSA(normal) - TSA(swap) over the scenes present in both files; bootstrap CI over scenes."""
    pn, ps = evalkit.pair_table(normal), evalkit.pair_table(swap)
    keys = sorted(set(pn) & set(ps))
    if not keys:
        return None
    tsa = lambda p: 0.5 * (float(p["plus"]["target_correct"]) + float(p["minus"]["target_correct"]))
    d = [tsa(pn[k]) - tsa(ps[k]) for k in keys]
    return evalkit.bootstrap_ci(d), len(keys)


def build(split):
    rows, md = [], []
    for tag, data, cond in ARMS:
        recs = load(f"results/eval/bc_{tag}__{split}.jsonl")
        if not recs:
            continue
        s = evalkit.summarize(recs)["all"]
        swp = load(f"results/eval/bc_{tag}_swap__{split}.jsonl")
        sens = sensitivity(recs, swp) if swp else None
        sens_txt = f"{100*sens[0][0]:.0f} [{100*sens[0][1]:.0f}, {100*sens[0][2]:.0f}]" if sens else "--"
        rows.append((tag, data, cond, s, sens_txt))
        md.append(f"| {tag} | {data} | {cond} | {s['n_pairs']} | {pct(s['PTA_sel'])} | {pct(s['PTA_grasp'])} | {pct(s['PTA_appr'])} | "
                  f"{100*s['grasp_rate'][0]:.0f} | {100*s['TSA'][0]:.1f} | {100*s['wrong_target'][0]:.1f} | {100*s['success'][0]:.1f} | {sens_txt} |")
    return rows, md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="iid")
    a = ap.parse_args()
    rows, md = build(a.split)
    if not rows:
        print("no learned-policy results yet"); sys.exit(0)
    head = ("| arm | data | conditioning | pairs | PTA [95% CI] | PTA-grasp [95% CI] | PTA-appr [95% CI] | grasp rate | TSA | wrong | success | language sens. |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    text = head + "\n" + "\n".join(md)
    os.makedirs(OUT, exist_ok=True)
    tex = [f"{cond} & {data} & {pct(s['PTA_sel'])} & {pct(s['PTA_grasp'])} & {100*s['TSA'][0]:.1f} & {100*s['wrong_target'][0]:.1f} & "
           f"{100*s['success'][0]:.1f} & {sens}" for tag, data, cond, s, sens in rows]
    open(f"{OUT}/pilot_{a.split}.tex", "w", newline="\n").write(" \\\\\n".join(tex) + "\n")     # rows separated by \\ ; main.tex supplies the last one
    open(f"results/eval/pilot_{a.split}.md", "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text)
    print(f"\nwrote {OUT}/pilot_{a.split}.tex ({len(rows)} rows)")
