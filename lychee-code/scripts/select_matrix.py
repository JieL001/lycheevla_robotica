"""PTA of selection-track arms across all evaluation splits (Wilson 95% intervals; no bootstrap, so it is fast).
   python scripts/select_matrix.py [--arms ...] [--tex]   ->  prints a markdown matrix; with --tex also writes ../tables/select_splits.tex
Splits: iid, sal (saliency-reversed), occ, dens, lang, attr, dr (held-out visual jitter)."""
import argparse, json, os, sys

sys.path.insert(0, ".")
from lychee import evalkit

SPLITS = [("iid", "IID"), ("sal", "Sal.-rev."), ("occ", "Occl."), ("dens", "Dens."), ("lang", "Lang."), ("attr", "Attr."), ("dr", "Vis.-DR")]
DEFAULT = ["r0_rho00_film", "r0_rho90_film", "r1_film", "r1u_film", "r1_late", "r1_token", "sym_r0_rho00", "sym_r0_rho90", "sym_r1"]
LABEL = {"r0_rho00_film": r"R0, $\rho=0$", "r0_rho50_film": r"R0, $\rho=0.5$", "r0_rho90_film": r"R0, $\rho=0.9$", "r0_rho97_film": r"R0, $\rho=0.97$",
         "r0_rho99_film": r"R0, $\rho=0.99$", "r1_film": "R1", "r1u_film": "R1u", "r1_late": "R1, late fusion", "r1_token": "R1, command token",
         "sym_r0_rho00": r"R0, $\rho=0$, fruit table", "sym_r0_rho90": r"R0, $\rho=0.9$, fruit table", "sym_r1": "R1, fruit table"}


def pta(arm, split):
    path = f"results/eval/s_{arm}__{split}.jsonl"
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path)]
    pt = evalkit.pair_table(recs)
    k = sum(1 for v in pt.values() if v["plus"]["target_correct"] and v["minus"]["target_correct"])
    return evalkit.wilson_ci(k, len(pt)) + (len(pt),)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT)
    ap.add_argument("--tex", action="store_true")
    a = ap.parse_args()
    md = ["| arm | " + " | ".join(n for _, n in SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]
    tex = []
    for arm in a.arms:
        cells, tcells = [], []
        for sp, _ in SPLITS:
            r = pta(arm, sp)
            cells.append("-" if r is None else f"{100*r[0]:.1f} [{100*r[1]:.0f}, {100*r[2]:.0f}]")
            tcells.append("--" if r is None else f"{100*r[0]:.1f}")
        md.append(f"| {arm} | " + " | ".join(cells) + " |")
        tex.append(f"{LABEL.get(arm, arm)} & " + " & ".join(tcells))
    print("\n".join(md))
    if a.tex:
        os.makedirs("../tables", exist_ok=True)
        open("../tables/select_splits.tex", "w", newline="\n").write(" \\\\\n".join(tex) + "\n")
        print("wrote ../tables/select_splits.tex")
