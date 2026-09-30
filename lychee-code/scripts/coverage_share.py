"""How much of the dial's cost at rho = 0.9 is a consequence of the mix of command types?  Two controls that leave the correlation between commands and salience out:
  far     R0 with the 'farthest' commands cut to the 146 that the dial leaves (matches one change of the command mix), macros \\cs<Set>...
  matched R0 on the scenes of the natural set with one command per scene drawn at the frequencies of ALL command templates of the biased set but not steered towards any fruit
          (scripts/make_matched_dataset.py; matches the whole mix), macros \\ms<Set>...
share = (PTA[R0] - PTA[control]) / (PTA[R0] - PTA[R0.9]) on the same scenes, with a bootstrap over scenes (shared by the three arms) and over the training seeds of each arm
(independent).  Reported for the pilot and the fresh evaluation sets.
   python scripts/coverage_share.py [--boot 4000]  ->  results/eval/coverage_share.md and ../tables/coverage_macros.tex
   (\\{cs,ms}<Set>D|L|H = point estimate, 2.5 %, 97.5 % as a fraction; N = seeds of the control; T = cost of the control in points)"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import per_seed

SETS = [("sal", "Sal", "pilot saliency-reversed"), ("iid", "Iid", "pilot IID"), ("salf", "SalF", "fresh saliency-reversed"), ("iidf", "IidF", "fresh IID"),
        ("salb", "Both", "fresh sal-both")]                        # the dial costs nothing detectable on attribute-OOD scenes: a share of that cost is not defined
BASES = dict(zero="r0_rho00_film", nine="r0_rho90_film", far="r0_rho00_far146", matched="r0_matched90_film")
CONTROLS = [("far", "cs", "146 farthest"), ("matched", "ms", "mix-matched")]


def mat(base, split):
    ps = per_seed(base, split)
    if not ps:
        return None
    keys = sorted(set.intersection(*[set(d) for d in ps]))
    return np.array([[d[k] for k in keys] for d in ps]), keys


def main(n_boot):
    md, macros = ["| control | set | R0 - control (points) | R0 - R0.9 (points) | share | 95 % interval |", "|---|---|---|---|---|---|"], []
    for _, pre, _ in CONTROLS:
        macros += [f"\\providecommand{{\\{pre}{m}{p}}}{{??}}" for _, m, _ in SETS for p in ("D", "L", "H", "N", "T")]
        macros += [f"\\providecommand{{\\{pre}{p}}}{{??}}" for p in ("Min", "Max")]
    shares = {}
    for ckey, pre, cname in CONTROLS:
        for split, m, name in SETS:
            got = {k: mat(BASES[k], split) for k in ("zero", "nine", ckey)}
            if any(v is None for v in got.values()):
                continue
            common = sorted(set.intersection(*[set(v[1]) for v in got.values()]))
            M = {k: np.array([[dict(zip(v[1], row))[c] for c in common] for row in v[0]]) for k, v in got.items()}
            rng = np.random.RandomState(0)
            Z, N9, F = M["zero"], M["nine"], M[ckey]
            est_c, est_d = (Z.mean() - F.mean()), (Z.mean() - N9.mean())
            boots = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.randint(0, len(common), len(common))
                zs, ns, fs = (rng.randint(0, len(X), len(X)) for X in (Z, N9, F))
                c, d = Z[zs][:, idx].mean() - F[fs][:, idx].mean(), Z[zs][:, idx].mean() - N9[ns][:, idx].mean()
                boots[i] = c / d if abs(d) > 1e-9 else np.nan
            lo, hi = np.nanquantile(boots, [0.025, 0.975])
            share = est_c / est_d
            shares.setdefault(pre, []).append(share)
            md.append(f"| {cname} | {name} (seeds {len(Z)}/{len(N9)}/{len(F)}) | {100*est_c:+.1f} | {100*est_d:+.1f} | {share:.2f} | [{lo:.2f}, {hi:.2f}] |")
            macros.append(f"\\renewcommand{{\\{pre}{m}D}}{{{share:.2f}}}\\renewcommand{{\\{pre}{m}L}}{{{lo:.2f}}}\\renewcommand{{\\{pre}{m}H}}{{{hi:.2f}}}"
                          f"\\renewcommand{{\\{pre}{m}N}}{{{len(F)}}}\\renewcommand{{\\{pre}{m}T}}{{{100*est_c:.1f}}}")
    for pre, v in shares.items():                                        # range of the point estimates over the evaluation sets (the two controls are reported separately)
        macros.append(f"\\renewcommand{{\\{pre}Min}}{{{min(v):.2f}}}\\renewcommand{{\\{pre}Max}}{{{max(v):.2f}}}")
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/coverage_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/coverage_share.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    main(ap.parse_args().boot)
