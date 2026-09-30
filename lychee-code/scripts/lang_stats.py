"""Benchmark-design statistics for the symbolic instruction layer (no simulator needed).

For each candidate count N: how often does a random scene admit (a) >=1 valid unique-target
instruction per family, (b) >=1 same-scene counterfactual pair, and (c) how well do language-blind
salience heuristics (most visible / frontmost fruit) do on the valid instructions?
"""
import json, sys, time
from collections import Counter
import numpy as np

sys.path.insert(0, ".")
from lychee.lang import valid_specs, counterfactual_pairs, FAMILIES
from lychee.scene import sample_fruits

N_SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
MIN_VIS = 0.2
rng = np.random.default_rng(0)
rows = {}
t0 = time.time()
for n in range(3, 17):
    fam_hit = Counter(); pair_hit = 0; n_specs = []; n_pairs = []; field = Counter()
    h_vis = h_front = h_vis_mat = tot = 0
    for _ in range(N_SCENES):
        fr = sample_fruits(rng, n)
        vs = valid_specs(fr, min_vis=MIN_VIS)
        for f in {s.family for s, _ in vs}:
            fam_hit[f] += 1
        n_specs.append(len(vs))
        pairs = counterfactual_pairs(fr, min_vis=MIN_VIS)
        n_pairs.append(len(pairs)); pair_hit += bool(pairs)
        for *_, f in pairs:
            field[f] += 1
        most_vis = max(fr, key=lambda f: f.vis).idx
        front = min(fr, key=lambda f: f.depth).idx
        for s, t in vs:
            tot += 1
            h_vis += (t == most_vis); h_front += (t == front)
            best = max((f for f in fr if f.mat == s.mat), key=lambda f: f.vis).idx
            h_vis_mat += (t == best)
    rows[n] = dict(
        p_any=float(np.mean([k > 0 for k in n_specs])),
        p_family={f: fam_hit[f] / N_SCENES for f in FAMILIES},
        p_pair=pair_hit / N_SCENES,
        mean_specs=float(np.mean(n_specs)), mean_pairs=float(np.mean(n_pairs)),
        edited_field_share={k: v / max(1, sum(field.values())) for k, v in field.items()},
        blind_most_visible=h_vis / tot, blind_frontmost=h_front / tot,
        blind_most_visible_given_maturity=h_vis_mat / tot,
    )
print(f"{N_SCENES} scenes per N, min_vis={MIN_VIS}, {time.time()-t0:.0f}s")
print(f"{'N':>2} {'P(any)':>7} {'P(pair)':>8} {'specs':>6} {'pairs':>6} | "
      + " ".join(f"{f:>11}" for f in FAMILIES) + " | blind: vis  front  vis|mat")
for n, r in rows.items():
    print(f"{n:>2} {r['p_any']:>7.3f} {r['p_pair']:>8.3f} {r['mean_specs']:>6.1f} {r['mean_pairs']:>6.1f} | "
          + " ".join(f"{r['p_family'][f]:>11.3f}" for f in FAMILIES)
          + f" | {r['blind_most_visible']:.3f} {r['blind_frontmost']:.3f} {r['blind_most_visible_given_maturity']:.3f}")
json.dump(rows, open("results/lang_stats.json", "w"), indent=1)
