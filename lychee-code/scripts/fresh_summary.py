"""Summary of the re-evaluation on FRESH scenes (rendered after the pilot; scripts/queue/lane_select_fresh.sh) and on the saliency-reversed split in which
both commands of a pair avoid the most salient fruit.
   python scripts/fresh_summary.py [--boot 4000]
Sets: iidf = 600 fresh IID scenes, salf = 600 fresh saliency-reversed scenes (first member constrained, as in the pilot split), salb = 600 scenes of the
split whose two commands both avoid the most salient fruit, attrf = 300 fresh attribute-OOD scenes.  Records: results/eval/s_<arm>[_s1|_s2]__<set>.jsonl.
Writes ../tables/fresh_main.tex (arms x sets), ../tables/fresh_compare.tex (comparisons, bootstrap over seeds and scenes, seed-level Welch), ../tables/fresh_macros.tex
(\\fm<Arm><Set> seed mean, \\fz<Arm><Set> seed 0, \\fa/\\fl/\\fh<Comparison> size and bootstrap interval, \\ft/\\fu<Comparison> ends of the seed-level interval)
and results/eval/fresh_summary.md."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from lychee import evalkit, stats
from select_seeds import per_seed, fmt, sgn, welch, REL

SETS = [("iidf", "IID", "Iid"), ("salf", "Sal.-rev.", "Sal"), ("salb", "Sal.-rev., both", "Both"), ("attrf", "Attr.", "Attr")]
ARMS = [
    ("Rzero", "r0_rho00_film", r"R0, $\rho=0$"),
    ("Rhalf", "r0_rho50_film", r"R0, $\rho=0.5$"),
    ("Rninety", "r0_rho90_film", r"R0, $\rho=0.9$"),
    ("Rninetyseven", "r0_rho97_film", r"R0, $\rho=0.97$"),
    ("Rninetynine", "r0_rho99_film", r"R0, $\rho=0.99$"),
    ("Rone", "r1_film", "R1"),
    ("Ronenu", "r1u_film", "R1u"),
    ("Rsame", "r1n_film", "R1n"),
    ("RzeroP", "r0p_film", "R0p"),
    ("Rlate", "r1_late", "R1, late fusion"),
    ("Rtoken", "r1_token", "R1, command token"),
    ("RzeroLate", "r0_rho00_late", r"R0, $\rho=0$, late fusion"),
    ("RninetyLate", "r0_rho90_late", r"R0, $\rho=0.9$, late fusion"),
    ("MixPaired", "mix_rho90_paired", r"R0, $\rho=0.9$ + paired scenes"),
    ("MixNatural", "mix_rho90_natural", r"R0, $\rho=0.9$ + relabelled scenes"),
    ("MixBiased", "mix_rho90_biased", r"R0, $\rho=0.9$ + biased scenes"),
    ("Far", "r0_rho00_far146", r"R0, $\rho=0$, 146 ``farthest''"),
    ("Rmatched", "r0_matched90_film", r"R0, command mix of $\rho=0.9$, no correlation"),
]
COMPARE = [  # label, macro, A, B, set
    (r"C1: R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, saliency-reversed", "ConeSal", "Rninety", "Rzero", "salf"),
    (r"C1: R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, IID", "ConeIid", "Rninety", "Rzero", "iidf"),
    (r"C2: R1 $-$ R0$_{\rho=0.9}$, saliency-reversed", "CtwoSal", "Rone", "Rninety", "salf"),
    (r"C2: R1 $-$ R0$_{\rho=0.9}$, IID", "CtwoIid", "Rone", "Rninety", "iidf"),
    (r"C3: R1 $-$ R1u, IID", "CthreeIid", "Rone", "Ronenu", "iidf"),
    (r"R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, both commands avoid the salient fruit", "ConeBoth", "Rninety", "Rzero", "salb"),
    (r"R0$_{\rho=0.97}$ $-$ R0$_{\rho=0}$, both commands avoid the salient fruit", "DialSevenBoth", "Rninetyseven", "Rzero", "salb"),
    (r"R0$_{\rho=0.99}$ $-$ R0$_{\rho=0}$, both commands avoid the salient fruit", "DialNineBoth", "Rninetynine", "Rzero", "salb"),
    (r"R1 $-$ R0$_{\rho=0}$, IID", "RoneZeroIid", "Rone", "Rzero", "iidf"),
    (r"R1 $-$ R0$_{\rho=0}$, saliency-reversed", "RoneZeroSal", "Rone", "Rzero", "salf"),
    (r"R1 $-$ R0$_{\rho=0}$, attribute-OOD", "RoneZeroAttr", "Rone", "Rzero", "attrf"),
    (r"R1n $-$ R0p, attribute-OOD", "SamePAttr", "Rsame", "RzeroP", "attrf"),
    (r"R1n $-$ R0p, IID", "SamePIid", "Rsame", "RzeroP", "iidf"),
    (r"R1u $-$ R0$_{\rho=0}$, attribute-OOD", "RnuZeroAttr", "Ronenu", "Rzero", "attrf"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$ + relabelled, saliency-reversed", "MixSal", "MixPaired", "MixNatural", "salf"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$ + relabelled, both commands avoid the salient fruit", "MixBoth", "MixPaired", "MixNatural", "salb"),
    (r"R0$_{\rho=0}$ with 146 ``farthest'' $-$ R0$_{\rho=0}$, saliency-reversed", "FarSal", "Far", "Rzero", "salf"),
    (r"R0$_{\rho=0.9}$ $-$ R0$_{\rho=0.9}$ with 146 ``farthest'' (natural), saliency-reversed", "NineFarSal", "Rninety", "Far", "salf"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$, saliency-reversed", "MatchedZeroSal", "Rmatched", "Rzero", "salf"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$, IID", "MatchedZeroIid", "Rmatched", "Rzero", "iidf"),
    (r"R0$_{\rho=0.9}$ $-$ R0 with the command mix of $\rho=0.9$, no correlation, saliency-reversed", "NineMatchedSal", "Rninety", "Rmatched", "salf"),
    (r"R0$_{\rho=0.9}$ $-$ R0 with the command mix of $\rho=0.9$, no correlation, IID", "NineMatchedIid", "Rninety", "Rmatched", "iidf"),
    (r"R0$_{\rho=0.9}$ + biased $-$ R0$_{\rho=0.9}$, saliency-reversed", "MixBiasedNineSal", "MixBiased", "Rninety", "salf"),
    (r"R0$_{\rho=0.9}$ + biased $-$ R0$_{\rho=0.9}$, IID", "MixBiasedNineIid", "MixBiased", "Rninety", "iidf"),
    (r"R0$_{\rho=0.9}$ + relabelled $-$ R0$_{\rho=0.9}$ + biased, saliency-reversed", "MixNaturalBiasedSal", "MixNatural", "MixBiased", "salf"),
    (r"R0$_{\rho=0.9}$ + relabelled $-$ R0$_{\rho=0.9}$ + biased, IID", "MixNaturalBiasedIid", "MixNatural", "MixBiased", "iidf"),
]
N_PLANNED = 5


def main(n_boot):
    data = {(k, s): per_seed(base, s) for k, base, _ in ARMS for s, _, _ in SETS}
    rel = {(k, s): per_seed(base, s, REL) for k, base, _ in ARMS for s in ("iidf", "salf", "salb")}
    macros, md, tex = [], ["| arm | seeds | " + " | ".join(n for _, n, _ in SETS) + " |", "|---|---|" + "---|" * len(SETS)], []
    macros += [f"\\providecommand{{\\fm{k}{m}}}{{??}}\\providecommand{{\\fz{k}{m}}}{{??}}" for k, _, _ in ARMS for _, _, m in SETS]
    macros += [f"\\providecommand{{\\{p}{m}}}{{??}}" for _, m, _, _, _ in COMPARE for p in ("fa", "fl", "fh", "fd", "ft", "fu", "fr", "fs", "fq")]
    for k, base, label in ARMS:
        cells = {}
        n_seeds = max(len(data[(k, s)]) for s, _, _ in SETS)
        if n_seeds == 0:
            continue
        for s, _, m in SETS:
            ps = data[(k, s)]
            if not ps:
                cells[s] = "--"
                continue
            means = [float(np.mean(list(d.values()))) for d in ps]
            rng_txt = f" ({fmt(min(means))}--{fmt(max(means))})" if len(ps) > 1 else ""
            cells[s] = f"{fmt(np.mean(means))}{rng_txt}"
            macros.append(f"\\renewcommand{{\\fm{k}{m}}}{{{fmt(np.mean(means))}}}\\renewcommand{{\\fz{k}{m}}}{{{fmt(means[0])}}}")
        md.append(f"| {k} | {n_seeds} | " + " | ".join(cells[s] for s, _, _ in SETS) + " |")
        rcell = {}
        for s in ("iidf", "salf"):                                    # depth and ordinal pairs only (post hoc), mean over the seeds and their range
            rps = rel[(k, s)]
            if rps:
                means = [float(np.mean(list(d.values()))) for d in rps]
                rcell[s] = f"{fmt(np.mean(means))}" + (f" ({fmt(min(means))}--{fmt(max(means))})" if len(rps) > 1 else "")
            else:
                rcell[s] = "--"
        tex.append(f"{label} & {n_seeds} & {cells['iidf']} & {rcell['iidf']} & {cells['salf']} & {rcell['salf']} & {cells['salb']} & {cells['attrf']}")
    rows, mdc = [], ["", "| comparison | seeds (A/B) | A - B [95% bootstrap] | relational pairs | seed-level t |", "|---|---|---|---|---|"]
    for i, (label, m, ka, kb, s) in enumerate(COMPARE):
        A, B = data[(ka, s)], data[(kb, s)]
        if not A or not B:
            continue
        d, lo, hi, ma, mb = stats.cluster_diff(A, B, n_boot=n_boot)
        RA, RB = rel.get((ka, s)), rel.get((kb, s))
        rtxt = rmd = "--"
        if RA and RB:
            rd, rlo, rhi, _, _ = stats.cluster_diff(RA, RB, n_boot=n_boot)
            rtxt = f"{sgn(rd)} [{sgn(rlo)}, {sgn(rhi)}]"; rmd = f"{100*rd:+.1f} [{100*rlo:+.1f}, {100*rhi:+.1f}]"
            macros.append(f"\\renewcommand{{\\fr{m}}}{{{sgn(rd)}}}\\renewcommand{{\\fs{m}}}{{{sgn(rlo)}}}\\renewcommand{{\\fq{m}}}{{{sgn(rhi)}}}")
        w = welch(ma, mb)
        ttxt = tmd = "--"
        if w:
            ttxt = f"[{sgn(w[0])}, {sgn(w[1])}]"; tmd = f"[{100*w[0]:+.1f}, {100*w[1]:+.1f}]"
            macros.append(f"\\renewcommand{{\\ft{m}}}{{{sgn(w[0])}}}\\renewcommand{{\\fu{m}}}{{{sgn(w[1])}}}")
        rows.append((i >= N_PLANNED, f"{label} & {len(A)}/{len(B)} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & {rtxt} & {ttxt}"))
        mdc.append(f"| {label} | {len(A)}/{len(B)} | {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | {rmd} | {tmd} |")
        macros.append(f"\\renewcommand{{\\fd{m}}}{{{sgn(d)}}}\\renewcommand{{\\fl{m}}}{{{sgn(lo)}}}\\renewcommand{{\\fh{m}}}{{{sgn(hi)}}}\\renewcommand{{\\fa{m}}}{{{100*abs(d):.1f}}}")
    # ---- pilot scenes against fresh scenes (the same checkpoints)
    macros += [f"\\providecommand{{\\{n}}}{{??}}" for n in ("frDropMin", "frDropMax", "frArms", "frSalLePilot", "frSalGtFresh", "fdidFD", "fdidFL", "fdidFH", "fdidBD", "fdidBL", "fdidBH",
                                                            "frRelP", "frRelF", "frNonRelP", "frNonRelF", "frRelN", "frRelPN")]
    drops, le_p, gt_f, n_arm = [], 0, 0, 0
    for k, base, _ in ARMS:
        pi, ps_ = per_seed(base, "iid"), per_seed(base, "sal")
        fi, fs_ = data[(k, "iidf")], data[(k, "salf")]
        if not (pi and ps_ and fi and fs_):
            continue
        mi, ms, fi_m, fs_m = (float(np.mean([np.mean(list(d.values())) for d in x])) for x in (pi, ps_, fi, fs_))
        drops.append(100 * (mi - fi_m)); n_arm += 1
        le_p += ms <= mi; gt_f += fs_m > fi_m
    if drops:
        macros.append(f"\\renewcommand{{\\frDropMin}}{{{min(drops):.1f}}}\\renewcommand{{\\frDropMax}}{{{max(drops):.1f}}}\\renewcommand{{\\frArms}}{{{n_arm}}}"
                      f"\\renewcommand{{\\frSalLePilot}}{{{le_p}}}\\renewcommand{{\\frSalGtFresh}}{{{gt_f}}}")
        print(f"pilot minus fresh IID over {n_arm} arms: {min(drops):.1f} to {max(drops):.1f} points; saliency-reversed <= IID: pilot {le_p}/{n_arm}, fresh > IID: {gt_f}/{n_arm}")
    # ---- difference in differences of the dial's cost, saliency-reversed (or sal-both) minus IID, on the fresh sets
    for tag, sname in (("F", "salf"), ("B", "salb")):
        if all(data[(k, s)] for k in ("Rninety", "Rzero") for s in (sname, "iidf")):
            d, lo, hi = stats.cluster_did(data[("Rninety", sname)], data[("Rninety", "iidf")], data[("Rzero", sname)], data[("Rzero", "iidf")], n_boot=n_boot)
            macros.append(f"\\renewcommand{{\\fdid{tag}D}}{{{sgn(d)}}}\\renewcommand{{\\fdid{tag}L}}{{{sgn(lo)}}}\\renewcommand{{\\fdid{tag}H}}{{{sgn(hi)}}}")
            mdc.append(f"| difference in differences of the dial's cost ({sname} minus iidf) | | {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | | |")
    # ---- what differs between the pilot and the fresh IID sample (R1, seed 0)
    try:
        fam = {}
        for tag, sp in (("P", "iid"), ("F", "iidf")):
            recs = [json.loads(l) for l in open(f"results/eval/s_r1_film__{sp}.jsonl")]
            f = evalkit.summarize(recs, by="family")
            n = sum(v["n_pairs"] for v in f.values())
            rel_n = sum(f[k]["n_pairs"] for k in ("mat_depth", "mat_ordinal"))
            non = [k for k in f if k not in ("mat_depth", "mat_ordinal")]
            hit = sum(f[k]["n_pairs"] * f[k]["PTA_sel"][0] for k in non) / max(1, sum(f[k]["n_pairs"] for k in non))
            fam[tag] = (rel_n, n, hit)
        macros.append(f"\\renewcommand{{\\frRelP}}{{{fam['P'][0]}}}\\renewcommand{{\\frRelF}}{{{fam['F'][0]}}}\\renewcommand{{\\frRelN}}{{{fam['F'][1]}}}"
                      f"\\renewcommand{{\\frNonRelP}}{{{100*fam['P'][2]:.1f}}}\\renewcommand{{\\frNonRelF}}{{{100*fam['F'][2]:.1f}}}")
        print(f"relational pairs: pilot {fam['P'][0]}/{fam['P'][1]}, fresh {fam['F'][0]}/{fam['F'][1]}; non-relational PTA (R1 seed 0): pilot {100*fam['P'][2]:.1f}, fresh {100*fam['F'][2]:.1f}")
    except Exception as e:
        print("family facts skipped:", e)
    body = ""
    for j, (explor, r) in enumerate(rows):
        if explor and (j == 0 or not rows[j - 1][0]):
            body += "\\midrule\n\\multicolumn{5}{@{}l}{\\emph{Exploratory}}\\\\\n"
        body += r + (" \\\\\n" if j < len(rows) - 1 else "\n")
    print("\n".join(md)); print("\n".join(mdc))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/fresh_main.tex", "w", newline="\n").write(" \\\\\n".join(tex) + "\n")
    open("../tables/fresh_compare.tex", "w", newline="\n").write(body)
    open("../tables/fresh_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/fresh_summary.md", "w", encoding="utf-8", newline="\n").write("\n".join(md + mdc) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    main(ap.parse_args().boot)
