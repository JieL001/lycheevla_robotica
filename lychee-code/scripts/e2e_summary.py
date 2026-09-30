"""End-to-end track: counts of the closed-loop episodes of the compact behaviour-cloning policy (40 IID scenes, 80 episodes) and the language probes.
   python scripts/e2e_summary.py   ->  ../tables/e2e_table.tex (rows of the table), ../tables/e2e_macros.tex, results/eval/e2e_summary.md
Rows: 8 epochs, 60 epochs, and (when the records exist) the 60-epoch policy with a blank command. (A run with the command of the OTHER member of the pair was dropped: it repeats the same
(scene, command) episodes with the two records exchanged; the pair statistics DiffFirst / OneAppr below carry the command-dependence information.)
Columns: episodes in which the fingertips came within 5 cm of a fruit (approached), gripped one, detached one, harvested the commanded fruit, and in brackets the number of them in which
the FIRST fruit approached / gripped / detached is a target of the (original) command; PTA and PTA-grasp count pairs whose two commands both lead to a target.
For every event the expected number of target hits under a uniformly random choice among the fruit of the scene is sum_i |T_i| / n_i, and the exact one-sided tail is the
Poisson-binomial probability of at least the observed number of hits.
Macros: \\ee<Row><Quantity> with Row in {Eight, Sixty, Blank}; DiffFirst = pairs in which both commands lead to an approach and the first fruit approached differs, OneAppr = pairs in which exactly one command leads to an approach."""
import json, os

import numpy as np

RECORDS = {"Eight": ("results/eval/bc_r1_film__iid.jsonl", "8 epochs"),
           "Sixty": ("results/eval/bc_r1_film_long__iid.jsonl", "60 epochs"),
           "Blank": ("results/eval/bc_r1_film_long_blank__iid.jsonl", "60 epochs, blank command")}
EVENTS = (("Appr", "first_approached"), ("Grip", "first_grasped"), ("Det", "first_detached"))


def pb_sf(ps, k):
    """P(X >= k) for a Poisson-binomial variable with success probabilities ps."""
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum())


def main():
    rows, macros, md = [], [], ["| policy | approached (target) | gripped (target) | detached (target) | harvested | PTA | PTA-grasp | pairs with both approached / both a target first |", "|---|---|---|---|---|---|---|---|"]
    for key, (path, label) in RECORDS.items():
        if not os.path.exists(path):
            macros.append(f"\\providecommand{{\\ee{key}Done}}{{0}}")
            continue
        recs = [json.loads(l) for l in open(path)]
        by = {}
        for r in recs:
            by.setdefault(r["index"], {})[r["which"]] = r
        pairs = [v for v in by.values() if "plus" in v and "minus" in v]
        cells, m = [], {}
        for ev, field in EVENTS:
            got = [r for r in recs if r[field] is not None and r[field] >= 0]
            hit = sum(r[field] in r["targets"] for r in got)
            ps = [len(r["targets"]) / r["n_fruit"] for r in got]
            m[ev] = (len(got), hit, sum(ps), pb_sf(ps, hit) if got else float("nan"))
            cells.append(f"{len(got)} ({hit})")
        one_appr = sum((v["plus"]["first_approached"] is not None and v["plus"]["first_approached"] >= 0) != (v["minus"]["first_approached"] is not None and v["minus"]["first_approached"] >= 0) for v in pairs)
        appr_both = [v for v in pairs if all(v[w]["first_approached"] is not None and v[w]["first_approached"] >= 0 for w in ("plus", "minus"))]
        diff_first = sum(v["plus"]["first_approached"] != v["minus"]["first_approached"] for v in appr_both)
        harvested = sum(bool(r["harvested"]) for r in recs)
        pta = sum(all(p[w]["first_detached"] in p[w]["targets"] and p[w]["first_detached"] >= 0 for w in ("plus", "minus")) for p in pairs)
        ptag = sum(all(p[w]["first_grasped"] in p[w]["targets"] and p[w]["first_grasped"] >= 0 for w in ("plus", "minus")) for p in pairs)
        both = {}
        for ev, field in EVENTS:
            ok = [p for p in pairs if all(p[w][field] is not None and p[w][field] >= 0 for w in ("plus", "minus"))]
            both[ev] = (len(ok), sum(all(p[w][field] in p[w]["targets"] for w in ("plus", "minus")) for p in ok))
        rows.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & {harvested} & {pta}/{len(pairs)} & {ptag}/{len(pairs)}")
        md.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {harvested} | {pta}/{len(pairs)} | {ptag}/{len(pairs)} | {both['Appr'][1]}/{both['Appr'][0]}; first fruit differs {diff_first}/{len(appr_both)}; approach under one command only {one_appr}/{len(pairs)} |")
        macros.append(f"\\providecommand{{\\ee{key}Done}}{{1}}")
        macros.append(f"\\providecommand{{\\ee{key}OneAppr}}{{{one_appr}}}\\providecommand{{\\ee{key}DiffFirst}}{{{diff_first}}}")
        for ev, _ in EVENTS:
            n, hit, exp, p = m[ev]
            macros.append(f"\\providecommand{{\\ee{key}{ev}N}}{{{n}}}\\providecommand{{\\ee{key}{ev}T}}{{{hit}}}\\providecommand{{\\ee{key}{ev}E}}{{{exp:.1f}}}"
                          f"\\providecommand{{\\ee{key}{ev}P}}{{{p:.4f}}}")
        macros.append(f"\\providecommand{{\\ee{key}Harv}}{{{harvested}}}\\providecommand{{\\ee{key}Pairs}}{{{len(pairs)}}}\\providecommand{{\\ee{key}Pta}}{{{pta}}}\\providecommand{{\\ee{key}PtaGrasp}}{{{ptag}}}")
        macros.append(f"\\providecommand{{\\ee{key}BothDet}}{{{both['Det'][0]}}}\\providecommand{{\\ee{key}BothGrip}}{{{both['Grip'][0]}}}"
                      f"\\providecommand{{\\ee{key}BothAppr}}{{{both['Appr'][0]}}}\\providecommand{{\\ee{key}BothApprT}}{{{both['Appr'][1]}}}")
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/e2e_table.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/e2e_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/e2e_summary.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
