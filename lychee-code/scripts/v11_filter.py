"""How much does the benchmark-v1.0 ordinal rule (which counts fruit of any visibility) matter?  A pair is "valid under the v1.1 rule" if every ordinal command in it has all
fruit of the commanded maturity up to position k (counted from the named side) at least 0.2 visible, so that a person who counts the fruit they can see reaches the label.
PTA is recomputed on the valid pairs only and compared with PTA on all pairs (no retraining, no simulation).
   python scripts/v11_filter.py   ->  results/eval/v11_filter.md, ../tables/v11_macros.tex (\\vv<Split>Share, \\vv<Arm><Split>Delta)"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, ".")
from lychee.paths import eval_meta                       # the meta.npz of the evaluation sets ship with the repository (data_meta/)
SPLITS = [("iid", "iid_600"), ("sal", "sal_600"), ("occ", "occ_300"), ("dens", "dens_300"), ("attr", "attr_300")]
ARMS = [("Rzero", "r0_rho00_film"), ("Rninety", "r0_rho90_film"), ("Rone", "r1_film"), ("Ronenu", "r1u_film")]
ORD = {"second": 2, "third": 3}


def valid_pairs(ev):
    d = np.load(eval_meta(ev))
    good = {}
    n_ord = n_bad = 0
    for i in np.flatnonzero(d["ok"]):
        okp = True
        for k in range(int(d["ncmd"][i])):
            if int(d["family"][i]) != 4:                     # 4 = mat_ordinal in the FAMS order of render_select.py
                continue
            text = str(d["text"][i, k])
            side = "left" if " left" in text else "right"
            kk = next((v for w, v in ORD.items() if w in text), 2)
            t = int(np.flatnonzero(d["tmask"][i, k])[0])
            m = d["fmat"][i, t]
            cand = [j for j in range(16) if d["fvalid"][i, j] and d["fmat"][i, j] == m]
            cand.sort(key=lambda j: d["fxy"][i, j, 0], reverse=(side == "right"))
            n_ord += 1
            if min(d["fvis"][i, j] for j in cand[:kk]) < 0.2:
                okp = False; n_bad += 1
        good[int(d["index"][i])] = okp
    return good, n_ord, n_bad


def pta(path, valid=None):
    n = k = 0
    pairs = {}
    for l in open(path):
        r = json.loads(l)
        pairs.setdefault(r["index"], {})[r["which"]] = r["target_correct"]
    for i, v in pairs.items():
        if len(v) < 2 or (valid is not None and not valid.get(i, True)):
            continue
        n += 1; k += bool(v["plus"] and v["minus"])
    return 100 * k / max(n, 1), n


def main():
    md, macros = ["| split | ordinal commands | violating the v1.1 rule | pairs valid (%) | arm | PTA all pairs | PTA valid pairs | delta |", "|---|---|---|---|---|---|---|---|"], []
    for sp, ev in SPLITS:
        valid, n_ord, n_bad = valid_pairs(ev)
        share = 100 * sum(valid.values()) / len(valid)
        macros.append(f"\\providecommand{{\\vv{sp.capitalize()}Share}}{{??}}\\renewcommand{{\\vv{sp.capitalize()}Share}}{{{100-share:.0f}}}")      # share of PAIRS that violate the rule
        macros.append(f"\\providecommand{{\\vv{sp.capitalize()}Ord}}{{??}}\\renewcommand{{\\vv{sp.capitalize()}Ord}}{{{100*n_bad/max(n_ord, 1):.1f}}}")   # share of ORDINAL COMMANDS that violate it
        deltas = []
        for key, base in ARMS:
            p = f"results/eval/s_{base}__{sp}.jsonl"
            if not os.path.exists(p):
                continue
            a, na = pta(p)
            b, nb = pta(p, valid)
            deltas.append(b - a)
            md.append(f"| {sp} | {n_ord} | {n_bad} ({100*n_bad/max(n_ord,1):.1f}%) | {share:.1f} | {key} | {a:.1f} ({na}) | {b:.1f} ({nb}) | {b-a:+.1f} |")
        if deltas:
            macros.append(f"\\providecommand{{\\vv{sp.capitalize()}MaxDelta}}{{??}}\\renewcommand{{\\vv{sp.capitalize()}MaxDelta}}{{{max(abs(x) for x in deltas):.1f}}}")
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/v11_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/v11_filter.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
