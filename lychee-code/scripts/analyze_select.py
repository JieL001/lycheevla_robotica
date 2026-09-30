"""Paired comparisons between selection-track arms (confirmatory list of the pre-registration draft + exploratory extras).
   python scripts/analyze_select.py [--tag_prefix s_] [--boot 10000]   ->  results/eval/select_comparisons.md
Each row: PTA (or another key) of arm A and arm B on the common scenes of a split, paired-bootstrap difference, McNemar exact p,
Holm-adjusted p within the confirmatory block and Benjamini-Hochberg-adjusted p within the exploratory block.
"""
import argparse, json, os, sys

sys.path.insert(0, ".")
from lychee import stats

CONFIRMATORY = [  # label, A, B, split, key
    ("C1  R0 rho=0.9 vs R0 rho=0, IID", "r0_rho90_film", "r0_rho00_film", "iid", "PTA_sel"),
    ("C1  R0 rho=0.9 vs R0 rho=0, saliency-reversed", "r0_rho90_film", "r0_rho00_film", "sal", "PTA_sel"),
    ("C2  R1 vs R0 rho=0.9, IID", "r1_film", "r0_rho90_film", "iid", "PTA_sel"),
    ("C2  R1 vs R0 rho=0.9, saliency-reversed", "r1_film", "r0_rho90_film", "sal", "PTA_sel"),
    ("C3  R1 vs R1u, IID", "r1_film", "r1u_film", "iid", "PTA_sel"),
]
EXPLORATORY = [
    ("R1 vs R0 rho=0, IID", "r1_film", "r0_rho00_film", "iid", "PTA_sel"),
    ("R1 vs R0 rho=0, saliency-reversed", "r1_film", "r0_rho00_film", "sal", "PTA_sel"),
    ("R1 vs R1u, saliency-reversed", "r1_film", "r1u_film", "sal", "PTA_sel"),
    ("R0 rho=0.5 vs R0 rho=0, IID", "r0_rho50_film", "r0_rho00_film", "iid", "PTA_sel"),
    ("R0 rho=0.9 vs R0 rho=0.5, saliency-reversed", "r0_rho90_film", "r0_rho50_film", "sal", "PTA_sel"),
    ("R1 vs R1 blank command, IID", "r1_film", "r1_blank_film", "iid", "PTA_sel"),
    ("R1 film vs R1 late fusion, IID", "r1_film", "r1_late", "iid", "PTA_sel"),
    ("R1 film vs R1 command token, IID", "r1_film", "r1_token", "iid", "PTA_sel"),
    ("R1 vs R0 rho=0.9, IID (TSA)", "r1_film", "r0_rho90_film", "iid", "TSA"),
]


def load(prefix, arm, split):
    path = f"results/eval/{prefix}{arm}__{split}.jsonl"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def block(rows, prefix, n_boot, adjust):
    out = []
    for label, A, B, split, key in rows:
        ra, rb = load(prefix, A, split), load(prefix, B, split)
        if not ra or not rb:
            continue
        r = stats.compare(stats.scene_outcomes(ra, key), stats.scene_outcomes(rb, key), n_boot=n_boot)
        out.append((label, key, r))
    if out:
        ps = [r["p_mcnemar"] if r["p_mcnemar"] is not None else 1.0 for _, _, r in out]
        adj = adjust(ps)
        return [(l, k, r, a) for (l, k, r), a in zip(out, adj)]
    return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_prefix", default="s_")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    lines = ["| comparison | key | scenes | A | B | A - B [95% CI] | McNemar p | adjusted p |", "|---|---|---|---|---|---|---|---|"]
    for name, rows, adjust in (("confirmatory (Holm)", CONFIRMATORY, stats.holm), ("exploratory (Benjamini-Hochberg)", EXPLORATORY, stats.bh)):
        res = block(rows, a.tag_prefix, a.boot, adjust)
        if not res:
            continue
        lines.append(f"| **{name}** | | | | | | | |")
        for label, key, r, adj in res:
            p = "--" if r["p_mcnemar"] is None else f"{r['p_mcnemar']:.2g}"
            lines.append(f"| {label} | {key} | {r['n']} | {100*r['mean_a']:.1f} | {100*r['mean_b']:.1f} | {100*r['diff']:+.1f} [{100*r['lo']:+.1f}, {100*r['hi']:+.1f}] | {p} | {adj:.2g} |")
    text = "\n".join(lines)
    print(text)
    os.makedirs("results/eval", exist_ok=True)
    open("results/eval/select_comparisons.md", "w", encoding="utf-8", newline="\n").write(text + "\n")
