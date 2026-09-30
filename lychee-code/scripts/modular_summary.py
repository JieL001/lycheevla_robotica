"""Modular baseline (fruit detector + parser + rules) against the learned selectors: PTA per command family on the IID scenes, PTA on the other sets, quality of the detector and the error
breakdown of the pipeline.
   python scripts/modular_summary.py
Reads the per-command records results/eval/s_mod_det[_s1|_s2]__<split>.jsonl (rules with the label margins), s_mod_free[...] (without the margins), s_mod_oracle__<split>.jsonl and the records
of the learned selectors, and results/eval/modular_stats.json (detector quality; written by scripts/eval_modular.py).  Writes ../tables/modular.tex (rows), ../tables/modular_errors.tex
(what happens to the commands on IID scenes, from THE SAME RECORDS as the PTA columns: every command is exactly one of correct / no selection / wrong fruit, so the shares add up to 100 %
and PTA >= 1 - 2 x (share of failed commands) holds per family; the script asserts it), ../tables/modular_macros.tex (\\mod<Quantity>) and results/eval/modular.md."""
import json, os, re, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import SEED_SUFFIXES

FAMS = [("mat_any", "any-of", "Any"), ("mat_side", "side", "Side"), ("mat_ordinal", "ordinal", "Ord"), ("mat_depth", "depth", "Depth"), ("mat", "unique", "Uniq")]
ROWS = [("R1, pixels", "r1_film", "Rone"), (r"R0 ($\rho=0$), pixels", "r0_rho00_film", "Rzero"),
        ("R1, fruit table", "sym_r1", "Sone"), ("Modular, benchmark margins", "mod_det", "Det"),
        ("Modular, no margins", "mod_free", "Free"), ("Modular, true table", "mod_oracle", "Oracle")]


def seeds(base, split):
    out = []
    for suf in SEED_SUFFIXES:
        p = f"results/eval/s_{base}{suf}__{split}.jsonl"
        if os.path.exists(p):
            out.append([json.loads(l) for l in open(p)])
    return out


def pair_table(recs):
    pairs = {}
    for r in recs:
        pairs.setdefault(r["index"], {})[r["which"]] = r
    return pairs


def pta(recs, fam=None):
    v = [all(x[w]["target_correct"] for w in ("plus", "minus")) for x in pair_table(recs).values() if "minus" in x and (fam is None or x["plus"]["family"] == fam)]
    return float(np.mean(v)) if v else float("nan")


def cell(vals, rng=False):
    """Mean over the training seeds (in percent); with rng the range of the seeds in brackets (only for the columns whose spread matters, to keep the table legible)."""
    return f"{100*np.mean(vals):.1f}" + (f" ({100*min(vals):.1f}--{100*max(vals):.1f})" if rng and len(vals) > 1 else "")


def outcomes(base, split="iid"):
    """Per command family, pooled over the seeds of ``base``: counts (commands, correct, no selection, wrong fruit) taken from the per-command records, and the mean PTA over the seeds.
    'no selection': the rules found no target among the detections or the pointing position was farther than the snap radius from every fruit (first_detached < 0)."""
    cnt = {f: [0, 0, 0, 0] for f, _, _ in FAMS}
    ptas = {f: [] for f, _, _ in FAMS}
    for recs in seeds(base, split):
        pairs = pair_table(recs)
        for f, _, _ in FAMS:
            ps = [v for v in pairs.values() if "minus" in v and v["plus"]["family"] == f]
            for v in ps:
                for w in ("plus", "minus"):
                    c = v[w]
                    cnt[f][0] += 1
                    if c["target_correct"]:
                        cnt[f][1] += 1
                    elif c["first_detached"] < 0:
                        cnt[f][2] += 1
                    else:
                        cnt[f][3] += 1
            if ps:
                ptas[f].append(float(np.mean([all(v[w]["target_correct"] for w in ("plus", "minus")) for v in ps])))
    return cnt, {f: (float(np.mean(v)) if v else float("nan")) for f, v in ptas.items()}


def check_outcomes(base, cnt, mean_pta):
    """The counting constraints between the per-command outcomes and PTA: a pair fails if either command fails, so PTA >= 1 - 2 e, and PTA <= min(TSA+, TSA-) <= pooled TSA."""
    for f, _, _ in FAMS:
        n, ok, none, wrong = cnt[f]
        if n == 0:
            continue
        assert ok + none + wrong == n
        e, p = (none + wrong) / n, mean_pta[f]
        assert p >= 1 - 2 * e - 1e-9, f"{base} {f}: PTA {p:.4f} is below the lower bound 1 - 2e = {1 - 2 * e:.4f}"
        assert p <= ok / n + 1e-9, f"{base} {f}: PTA {p:.4f} exceeds the accuracy per command {ok / n:.4f}"


def stat_keys(st, base, split):
    """Keys of modular_stats.json that belong to exactly this arm (its own name and its seeds _s<k>), not to arms whose name only starts with the same letters."""
    return [k for k in st if k == f"{base}__{split}" or re.fullmatch(re.escape(base) + r"_s\d+__" + split, k)]


def main():
    rows, md, macros = [], ["| row | seeds | IID | any-of | side | ordinal | depth | unique | Occl. no any-of | Attr. | IID fresh |", "|---|---|---|---|---|---|---|---|---|---|---|"], []
    for label, base, tag in ROWS:
        iid = seeds(base, "iid")
        if not iid:
            continue
        cells = [cell([pta(r) for r in iid], rng=True)]
        macros.append(f"\\providecommand{{\\mod{tag}Iid}}{{{100*np.mean([pta(r) for r in iid]):.1f}}}")
        for fam, _, ft in FAMS:
            v = [pta(r, fam) for r in iid]
            cells.append(cell(v)); macros.append(f"\\providecommand{{\\mod{tag}{ft}}}{{{100*np.mean(v):.1f}}}")
        occ = seeds(base, "occ")
        if occ:                                                                                # pooled over the pairs of the four families other than any-of, not the mean of family means
            v = [np.mean([all(y[w]["target_correct"] for w in ("plus", "minus")) for y in pair_table(r).values() if "minus" in y and y["plus"]["family"] != "mat_any"]) for r in occ]
            cells.append(cell(v)); macros.append(f"\\providecommand{{\\mod{tag}OccN}}{{{100*np.mean(v):.1f}}}")
        else:
            cells.append("--")
        for sp, tg in (("attr", "Attr"), ("iidf", "IidF")):
            s = seeds(base, sp)
            cells.append(cell([pta(r) for r in s], rng=(sp == "attr")) if s else "--")
            if s:
                macros.append(f"\\providecommand{{\\mod{tag}{tg}}}{{{100*np.mean([pta(r) for r in s]):.1f}}}")
        rows.append(f"{label} & {len(iid)} & " + " & ".join(cells))
        md.append(f"| {tag} | {len(iid)} | " + " | ".join(cells) + " |")
    macros.append(f"\\providecommand{{\\nsMod}}{{{len(seeds('mod_det', 'iid'))}}}")                  # number of detector seeds

    # detector quality (mean over the detector seeds; exact arm names, the oracle runs are not detector runs)
    st_path = "results/eval/modular_stats.json"
    st = json.load(open(st_path)) if os.path.exists(st_path) else {}
    for sp, sname in (("iid", "IID"), ("occ", "Occl")):
        ks = stat_keys(st, "mod_det", sp)
        for q in ("precision", "recall", "maturity_acc", "parsed"):
            if ks:
                macros.append(f"\\providecommand{{\\modDet{q.title().replace('_', '')}{sname}}}{{{100*np.mean([st[k][q] for k in ks]):.1f}}}")
    # the pointing position is never farther than the snap radius from every fruit (the evaluation log counts these commands separately from 'no target'): checked, not assumed
    far = sum(st[k]["errors"].get(f, [0, 0, 0, 0])[2] for base in ("mod_det", "mod_free") for k in stat_keys(st, base, "iid") for f, _, _ in FAMS)
    assert far == 0, f"{far} commands with a pointing position farther than the snap radius from every fruit; the caption of the error table says there are none"

    # what happens to the commands on IID scenes, both versions of the rules, from the same records as the PTA columns
    err_rows, md_err = [], ["| family | commands | benchmark: correct | no target | wrong | no margins: correct | no target | wrong |", "|---|---|---|---|---|---|---|---|"]
    strict, mp_s = outcomes("mod_det")
    free, mp_f = outcomes("mod_free")
    check_outcomes("mod_det", strict, mp_s)
    check_outcomes("mod_free", free, mp_f)
    for fam, name, ft in FAMS:
        s, f = strict[fam], free[fam]
        if s[0] == 0:
            continue
        assert s[0] == f[0], "both versions of the rules must have been evaluated on the same commands"
        sh = lambda c, i: 100 * c[i] / c[0]
        err_rows.append(f"{name} & {s[0]} & {sh(s, 1):.1f} & {sh(s, 2):.1f} & {sh(s, 3):.1f} & {sh(f, 1):.1f} & {sh(f, 2):.1f} & {sh(f, 3):.1f}")
        md_err.append("| " + " | ".join([name, str(s[0])] + [f"{sh(x, i):.1f}" for x in (s, f) for i in (1, 2, 3)]) + " |")
        macros.append(f"\\providecommand{{\\modErr{ft}None}}{{{sh(s, 2):.1f}}}\\providecommand{{\\modErr{ft}Wrong}}{{{sh(s, 3):.1f}}}\\providecommand{{\\modErr{ft}Correct}}{{{sh(s, 1):.1f}}}")
        macros.append(f"\\providecommand{{\\modFreeErr{ft}None}}{{{sh(f, 2):.1f}}}\\providecommand{{\\modFreeErr{ft}Wrong}}{{{sh(f, 3):.1f}}}\\providecommand{{\\modFreeErr{ft}Correct}}{{{sh(f, 1):.1f}}}")
    os.makedirs("../tables", exist_ok=True)
    open("../tables/modular.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/modular_errors.tex", "w", newline="\n").write(" \\\\\n".join(err_rows) + "\n")
    open("../tables/modular_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/modular.md", "w", encoding="utf-8", newline="\n").write("\n".join(md + [""] + md_err) + "\n")
    print("\n".join(md + [""] + md_err))


if __name__ == "__main__":
    main()
