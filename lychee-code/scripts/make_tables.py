"""Generate LaTeX table bodies for the manuscript from results/eval/*.jsonl (so numbers are never typed by hand).
   python scripts/make_tables.py
     -> ../tables/reference_iid.tex      scripted reference policies on the IID split (overall + per family)
     -> ../tables/reference_splits.tex   PTA of the reference policies on every evaluated split
     -> ../tables/reference_meta.tex     \\newcommand values quoted in the text (pairs, family sizes, chance level)
Pair-level rates carry Wilson 95% intervals (lychee.evalkit); per-command rates a bootstrap over scenes.
"""
import json, os, sys

sys.path.insert(0, ".")
from lychee import evalkit

OUT = "../tables"
os.makedirs(OUT, exist_ok=True)

ORDER = [("expert", "Oracle (the label)"),
         ("attr_visible", "Maturity word only"),
         ("rel_only", "Relation words only"),
         ("random", "Random fruit"),
         ("blind_visible", "Language-blind: most visible"),
         ("blind_front", "Language-blind: nearest")]
SPLITS = [("iid", "IID"), ("occ_ood", "Occl."), ("density_ood", "Dens."), ("lang_ood", "Lang."), ("attr_ood", "Attr."),
          ("saliency_rev", "Sal.-rev.")]
FAMS = [("mat_any", "any-of"), ("mat_side", "side"), ("mat_ordinal", "ordinal"), ("mat_depth", "depth")]


def load(tag, split):
    path = f"results/eval/{tag}_{split}.jsonl"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def ci(x, digits=1):
    m, lo, hi = x
    return f"{100*m:.{digits}f} [{100*lo:.0f}, {100*hi:.0f}]"


def reference_iid(split="iid"):
    rows, meta = [], {}
    ref = load("expert", split)
    fam_n = {k: v["n_pairs"] for k, v in evalkit.summarize(ref, by="family").items()}
    n_pairs = evalkit.summarize(ref)["all"]["n_pairs"]
    for tag, label in ORDER:
        recs = load(tag, split)
        if not recs:
            continue
        s = evalkit.summarize(recs)["all"]
        fam = evalkit.summarize(recs, by="family")
        macro = evalkit.macro_average(recs)[0]
        cells = [f"{100*fam[k]['PTA_sel'][0]:.0f}" if k in fam else "--" for k, _ in FAMS]
        rows.append(f"{label} & {ci(s['PTA_sel'])} & {100*macro:.0f} & {100*s['TSA'][0]:.1f} & {100*s['wrong_target'][0]:.1f} & "
                    f"{100*s['clean_success'][0]:.1f} & " + " & ".join(cells))
        if tag == "expert":
            meta["chance"] = 100 * s["chance_PTA"]
    counts = " & ".join(str(fam_n.get(k, 0)) for k, _ in FAMS)
    rows.append(f"\\emph{{pairs per family}} & {n_pairs} & & & & & {counts}")
    open(f"{OUT}/reference_{split}.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")   # rows are separated by \\ ; main.tex supplies the last one
    lines = [f"\\newcommand{{\\refPairs}}{{{n_pairs}}}", f"\\newcommand{{\\refChance}}{{{meta.get('chance', float('nan')):.0f}}}"]
    for k, name in FAMS:
        lines.append(f"\\newcommand{{\\refN{name.replace('-', '').capitalize()}}}{{{fam_n.get(k, 0)}}}")
    open(f"{OUT}/reference_meta.tex", "w", newline="\n").write("\n".join(lines) + "\n")
    print(f"wrote {OUT}/reference_{split}.tex ({len(rows)} rows, {n_pairs} pairs) and reference_meta.tex")


def reference_splits():
    rows = []
    for tag, label in ORDER:
        cells, any_ = [], False
        for sp, _ in SPLITS:
            recs = load(tag, sp)
            if not recs:
                cells.append("--")
                continue
            any_ = True
            s = evalkit.summarize(recs)["all"]
            cells.append(f"{ci(s['PTA_sel'])} ({s['n_pairs']})")
        if any_:
            rows.append(f"{label} & " + " & ".join(cells))
    open(f"{OUT}/reference_splits.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    print(f"wrote {OUT}/reference_splits.tex ({len(rows)} rows)")


if __name__ == "__main__":
    reference_iid("iid")
    reference_splits()
