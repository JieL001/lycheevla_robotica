"""How often can each command family form a same-scene counterfactual pair?  (numpy only)"""
import sys, json; sys.path.insert(0, ".")
import numpy as np
from lychee.layout import sample_layout
from lychee.lang import counterfactual_pairs_sets, FAMILIES

N_SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
out = {}
for label, n_range in [("N=3-5", (3, 5)), ("N=6-10", (6, 10)), ("N=11-16", (11, 16))]:
    hit = {f: 0 for f in FAMILIES}; any_hit = 0
    for seed in range(N_SCENES):
        lay = sample_layout(1_000_000 + seed, n_range=n_range)
        pairs = counterfactual_pairs_sets(lay.fruit_views(), min_vis=0.4)
        fams = {p[0].family for p in pairs}
        any_hit += bool(fams)
        for f in fams: hit[f] += 1
    out[label] = {f: hit[f] / N_SCENES for f in FAMILIES} | {"any": any_hit / N_SCENES}
    print(f"{label:8s} P(scene admits a pair):  any {any_hit/N_SCENES:.2f} | " + " ".join(f"{f}={hit[f]/N_SCENES:.2f}" for f in FAMILIES))
json.dump(out, open("results/family_feasibility.json", "w"), indent=1)
