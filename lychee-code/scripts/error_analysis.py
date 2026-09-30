"""Where do the wrong answers of the selectors go, and how does accuracy depend on whether the commanded fruit is the most salient one?  For every arm and for the
IID and saliency-reversed splits (pooled over the available training seeds): the share of commands answered with the most salient fruit, the share of wrong
answers that are the most salient fruit, the share of wrong answers that are invalid (no fruit within 20 px), and the accuracy on cue-ALIGNED commands (the target
set contains the most salient fruit) and on cue-ANTI-ALIGNED commands (it does not), the latter being the cleaner readout of a learned cue (pooled over both splits).
Salience = visibility x projected radius^2 (= visibility / depth^2 up to a constant).
   python scripts/error_analysis.py   ->  results/eval/error_analysis.md, ../tables/error_macros.tex (\\er<Arm><Split><Quantity>, \\ea<Arm>Aligned|Anti)"""
import json, os, sys

import numpy as np

sys.path.insert(0, ".")
from lychee.paths import eval_meta                       # the meta.npz of the evaluation sets ship with the repository (data_meta/)
SPLITS = {"iid": "iid_600", "sal": "sal_600"}
ARMS = [("Rzero", "r0_rho00_film"), ("Rhalf", "r0_rho50_film"), ("Rninety", "r0_rho90_film"), ("Rninetyseven", "r0_rho97_film"),
        ("Rninetynine", "r0_rho99_film"), ("Rone", "r1_film"), ("Ronenu", "r1u_film"), ("Rsame", "r1n_film"), ("RzeroP", "r0p_film"),
        ("RzeroLate", "r0_rho00_late"), ("RninetyLate", "r0_rho90_late"), ("RzeroToken", "r0_rho00_token"), ("RninetyToken", "r0_rho90_token"),
        ("Far", "r0_rho00_far146"), ("Rmatched", "r0_matched90_film"), ("MixPaired", "mix_rho90_paired"), ("MixNatural", "mix_rho90_natural")]


def top_of(split):
    d = np.load(eval_meta(SPLITS[split]))
    sal = d["fvis"] * d["frad"] ** 2 * d["fvalid"]
    return {int(i): int(sal[k].argmax()) for k, i in enumerate(d["index"]) if d["ok"][k]}


LABEL = {"Rzero": r"R0, $\rho=0$", "Rhalf": r"R0, $\rho=0.5$", "Rninety": r"R0, $\rho=0.9$", "Rninetyseven": r"R0, $\rho=0.97$", "Rninetynine": r"R0, $\rho=0.99$",
         "Rone": "R1", "Ronenu": "R1u", "Rsame": "R1n", "RzeroP": "R0p", "RzeroLate": r"R0, $\rho=0$, late fusion", "RninetyLate": r"R0, $\rho=0.9$, late fusion",
         "RzeroToken": r"R0, $\rho=0$, command token", "RninetyToken": r"R0, $\rho=0.9$, command token",
         "Far": r"R0, $\rho=0$, 146 ``farthest''", "Rmatched": r"R0, command mix of $\rho=0.9$, no correlation", "MixPaired": r"R0, $\rho=0.9$ + paired", "MixNatural": r"R0, $\rho=0.9$ + relabelled"}
CUE_ROWS = ["Rzero", "Rhalf", "Rninety", "Rninetyseven", "Rninetynine", "Rmatched", "Rone", "Ronenu", "Rsame"]


def main():
    md = ["| arm | split | seeds | commands | wrong (%) | selects most salient (%) | wrong that are most salient (%) | wrong that are invalid (%) |", "|---|---|---|---|---|---|---|---|"]
    md2 = ["", "| arm | seeds | aligned commands | accuracy aligned (%) | anti-aligned commands | accuracy anti-aligned (%) | gap (points) |", "|---|---|---|---|---|---|---|"]
    macros, cue_rows = [], []
    tops = {sp: top_of(sp) for sp in SPLITS}
    # how many test commands name the most salient fruit, per member of the pair (the saliency-reversed split constrains only the first member)
    for sp, ev in SPLITS.items():
        d = np.load(eval_meta(ev))
        tm, okm = d["tmask"], d["ok"].astype(bool)
        top = np.array([tops[sp].get(int(i), 0) for i in d["index"]])
        al = [tm[okm, k][np.arange(okm.sum()), top[okm]].mean() for k in (0, 1)]
        for k, nm in enumerate(("Plus", "Minus")):
            macros.append(f"\\providecommand{{\\al{sp.capitalize()}{nm}}}{{??}}\\renewcommand{{\\al{sp.capitalize()}{nm}}}{{{100*al[k]:.0f}}}")
        macros.append(f"\\providecommand{{\\al{sp.capitalize()}All}}{{??}}\\renewcommand{{\\al{sp.capitalize()}All}}{{{100*(al[0]+al[1])/2:.0f}}}")
        print(f"{sp}: share of commands naming the most salient fruit: first member {100*al[0]:.1f}%, second member {100*al[1]:.1f}%, all {100*(al[0]+al[1])/2:.1f}%")
    md3 = ["", "| arm | seeds | 'farthest' commands | accuracy (%) | 'nearest' commands | accuracy (%) |", "|---|---|---|---|---|---|"]
    for key, base in ARMS:
        al_n = al_ok = an_n = an_ok = ns_pool = 0
        far_n = far_ok = near_n = near_ok = 0                                    # accuracy on "farthest" / "nearest" depth commands (coverage control of the dial)
        fam_cnt = {f: [0, 0, 0, 0] for f in ("mat_ordinal", "mat_depth")}        # per relational family: aligned n, ok, anti-aligned n, ok
        for sp in SPLITS:
            n = wrong = top_sel = wrong_top = wrong_inv = ns = 0
            exp_top = 0.0                                                  # expected number of wrong answers that name the most salient fruit if wrong answers were uniform over the non-targets
            for suf in ("", "_s1", "_s2", "_s3", "_s4"):
                p = f"results/eval/s_{base}{suf}__{sp}.jsonl"
                if not os.path.exists(p):
                    continue
                ns += 1
                for l in open(p):
                    r = json.loads(l)
                    s, t = r["first_detached"], r["targets"]
                    top = tops[sp][r["index"]]
                    ok = s >= 0 and s in t
                    n += 1
                    top_sel += s >= 0 and s == top
                    if not ok:
                        wrong += 1
                        wrong_top += s >= 0 and s == top
                        exp_top += (top not in t) / max(1, r["n_fruit"] - len(t))
                        wrong_inv += s < 0
                    if top in t:
                        al_n += 1; al_ok += ok
                    else:
                        an_n += 1; an_ok += ok
                    if r["family"] == "mat_depth":
                        far_n += "farthest" in r["text"]; far_ok += ok and "farthest" in r["text"]
                        near_n += "nearest" in r["text"]; near_ok += ok and "nearest" in r["text"]
                    if r["family"] in fam_cnt:
                        c = fam_cnt[r["family"]]
                        if top in t:
                            c[0] += 1; c[1] += ok
                        else:
                            c[2] += 1; c[3] += ok
            if n == 0:
                continue
            ns_pool = max(ns_pool, ns)
            md.append(f"| {key} | {sp} | {ns} | {n} | {100*wrong/n:.1f} | {100*top_sel/n:.1f} | {100*wrong_top/max(wrong,1):.1f} | {100*wrong_inv/max(wrong,1):.1f} |")
            tag = f"{key}{sp.capitalize()}"
            for q, v in (("Wrong", 100 * wrong / n), ("TopSel", 100 * top_sel / n), ("WrongTop", 100 * wrong_top / max(wrong, 1)), ("WrongInv", 100 * wrong_inv / max(wrong, 1)),
                         ("ChanceTop", 100 * exp_top / max(wrong, 1))):
                macros.append(f"\\providecommand{{\\er{tag}{q}}}{{??}}\\renewcommand{{\\er{tag}{q}}}{{{v:.0f}}}")
        if far_n and near_n:
            md3.append(f"| {key} | {ns_pool} | {far_n} | {100*far_ok/far_n:.1f} | {near_n} | {100*near_ok/near_n:.1f} |")
            macros.append(f"\\providecommand{{\\ef{key}Far}}{{??}}\\renewcommand{{\\ef{key}Far}}{{{100*far_ok/far_n:.0f}}}"
                          f"\\providecommand{{\\ef{key}Near}}{{??}}\\renewcommand{{\\ef{key}Near}}{{{100*near_ok/near_n:.0f}}}"
                          f"\\providecommand{{\\ef{key}FarN}}{{??}}\\renewcommand{{\\ef{key}FarN}}{{{far_n}}}")
        if al_n and an_n:
            a, b = 100 * al_ok / al_n, 100 * an_ok / an_n
            md2.append(f"| {key} | {ns_pool} | {al_n} | {a:.1f} | {an_n} | {b:.1f} | {a-b:.1f} |")
            macros.append(f"\\providecommand{{\\ea{key}Aligned}}{{??}}\\renewcommand{{\\ea{key}Aligned}}{{{a:.1f}}}"
                          f"\\providecommand{{\\ea{key}Anti}}{{??}}\\renewcommand{{\\ea{key}Anti}}{{{b:.1f}}}"
                          f"\\providecommand{{\\ea{key}Gap}}{{??}}\\renewcommand{{\\ea{key}Gap}}{{{a-b:.1f}}}")
            cells = [f"{a:.1f}", f"{b:.1f}"]
            for fam, fname in (("mat_ordinal", "Ord"), ("mat_depth", "Dep")):
                c = fam_cnt[fam]
                fa, fb = (100 * c[1] / c[0] if c[0] else float("nan")), (100 * c[3] / c[2] if c[2] else float("nan"))
                cells += [f"{fa:.1f}", f"{fb:.1f}"]
                macros.append(f"\\providecommand{{\\ea{key}{fname}Aligned}}{{??}}\\renewcommand{{\\ea{key}{fname}Aligned}}{{{fa:.1f}}}"
                              f"\\providecommand{{\\ea{key}{fname}Anti}}{{??}}\\renewcommand{{\\ea{key}{fname}Anti}}{{{fb:.1f}}}")
            if key in CUE_ROWS:
                cue_rows.append(f"{LABEL[key]} & {ns_pool} & " + " & ".join(cells))
    print("\n".join(md + md2 + md3))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/cue_readout.tex", "w", newline="\n").write(" \\\\\n".join(cue_rows) + "\n")
    open("../tables/error_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/error_analysis.md", "w", encoding="utf-8", newline="\n").write("\n".join(md + md2 + md3) + "\n")


if __name__ == "__main__":
    main()
