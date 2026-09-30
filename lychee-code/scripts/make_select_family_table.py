"""Per-family, per-fruit-count and per-visibility PTA of selection-track arms (Wilson 95% intervals).
   python scripts/make_select_family_table.py [--split iid] [--arms r1_film r0_rho00_film r0_rho90_film sym_r1]
   ->  ../tables/select_family.tex (rows = arms, columns = families) and results/eval/select_family.md (all breakdowns)"""
import argparse, json, os, sys

sys.path.insert(0, ".")
from lychee import evalkit

FAMS = [("mat_any", "any-of"), ("mat_side", "side"), ("mat_ordinal", "ordinal"), ("mat_depth", "depth"), ("mat", "unique")]
LABEL = {"r1_film": "R1", "r0_rho00_film": r"R0, $\rho=0$", "r0_rho50_film": r"R0, $\rho=0.5$", "r0_rho90_film": r"R0, $\rho=0.9$",
         "r1u_film": "R1u", "r1_blank_film": "R1 blank", "sym_r1": "R1, fruit table", "sym_r0_rho00": r"R0, $\rho=0$, fruit table",
         "sym_r0_rho90": r"R0, $\rho=0.9$, fruit table", "r1n_film": "R1n", "r0p_film": "R0p", "r1_late": "R1, late fusion", "r1_token": "R1, command token",
         "r0_rho97_film": r"R0, $\rho=0.97$", "r0_rho99_film": r"R0, $\rho=0.99$",
         "mod_oracle": "Modular, true fruit table", "mod_det": "Modular, benchmark margins", "mod_free": "Modular, no margins",
         "r0_matched90_film": r"R0, command mix of $\rho=0.9$, no correlation", "r0_rho00_far146": r"R0, $\rho=0$, 146 ``farthest''"}


def load(arm, split):
    path = f"results/eval/s_{arm}__{split}.jsonl"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def ci(x):
    m, lo, hi = x
    return f"{100*m:.0f} [{100*lo:.0f}, {100*hi:.0f}]"


def binned(recs, key, fn):
    out = []
    for r in recs:
        r = dict(r); r[key] = fn(r); out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="iid")
    ap.add_argument("--arms", nargs="+", default=["r1_film", "r0_rho00_film", "r0_rho90_film", "r0_rho00_far146", "r0_matched90_film", "r1u_film", "r1n_film", "sym_r1"])
    a = ap.parse_args()
    FAMS = [f for f in FAMS if not (a.split == "attr" and f[0] == "mat")]        # the attribute-OOD split has no unique commands
    md, rows = [], []
    for arm in a.arms:
        recs = load(arm, a.split)
        if not recs:
            continue
        fam = evalkit.summarize(recs, by="family")
        cells = [ci(fam[k]["PTA_sel"]) if k in fam else "--" for k, _ in FAMS]
        rows.append(f"{LABEL.get(arm, arm)} & " + " & ".join(cells))
        md.append(f"### {arm} on {a.split}\n| group | pairs | PTA [95% CI] | TSA |\n|---|---|---|---|")
        for k, name in FAMS:
            if k in fam:
                md.append(f"| family {name} | {fam[k]['n_pairs']} | {ci(fam[k]['PTA_sel'])} | {100*fam[k]['TSA'][0]:.0f} |")
        for key, fn, order in (("nbin", lambda r: "3-5" if r["n_fruit"] <= 5 else ("6-8" if r["n_fruit"] <= 8 else "9-16"), ["3-5", "6-8", "9-16"]),
                               ("vbin", lambda r: "<0.7" if r["target_vis"] < 0.7 else ">=0.7", ["<0.7", ">=0.7"])):
            g = evalkit.summarize(binned(recs, key, fn), by=key)
            for b in order:
                if b in g:
                    md.append(f"| {'fruit count' if key == 'nbin' else 'target visibility'} {b} | {g[b]['n_pairs']} | {ci(g[b]['PTA_sel'])} | {100*g[b]['TSA'][0]:.0f} |")
        md.append("")
    if not rows:
        print("no results yet"); sys.exit(0)
    os.makedirs("../tables", exist_ok=True)
    fam_n = evalkit.summarize(load(a.arms[0], a.split), by="family") if load(a.arms[0], a.split) else {}
    rows.append("\\emph{pairs per family} & " + " & ".join(str(fam_n[k]["n_pairs"]) if k in fam_n else "--" for k, _ in FAMS))
    suffix = "" if a.split == "iid" else f"_{a.split}"
    open(f"../tables/select_family{suffix}.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open(f"results/eval/select_family{suffix}.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))
