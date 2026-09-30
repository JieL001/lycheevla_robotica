"""How strongly does the bias dial steer training commands?  For each bias-dial training set: the share of commands whose target set
contains the most salient fruit, the same restricted to unique-target commands, and the family mix.
   python scripts/bias_dial_stats.py 300   ->  results/bias_dial_stats.json"""
import collections, json, sys
sys.path.insert(0, ".")
import numpy as np
from dataclasses import replace
from lychee.splits import SPLITS, sample_config, layout_for, salience

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
res = {}
for name, sp in (("rho0", replace(SPLITS["train"], name="rho0", seed_base=100_000_000)), ("rho0.5", SPLITS["train_b50"]), ("rho0.9", SPLITS["train_b90"])):
    fam, hit, uhit, nu = collections.Counter(), 0, 0, 0
    for i in range(N):
        cfg = sample_config(sp, i, pair=False)
        fr = layout_for(sp, cfg.scene_seed).fruit_views()
        top = int(np.argmax([salience(f) for f in fr]))
        fam[cfg.plus.spec.family] += 1
        hit += top in cfg.plus.targets
        if len(cfg.plus.targets) == 1:
            nu += 1; uhit += top in cfg.plus.targets
    res[name] = dict(n=N, p_top_in_targets=hit / N, p_top_is_target_unique=uhit / max(1, nu), n_unique=nu, family_mix=dict(sorted(fam.items())))
    print(name, res[name], flush=True)
json.dump(res, open("results/bias_dial_stats.json", "w"), indent=1)
