import json, sys, collections, math
import numpy as np

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def load(p): return [json.loads(l) for l in open(p)]
sets = {"N=3-10": load("results/expert/n3_10.jsonl"), "N=11-16": load("results/expert/n11_16.jsonl")}
for name, rows in sets.items():
    valid = [r for r in rows if r["status"] != "no_valid_instruction"]
    ok = sum(r["status"] == "ok" for r in valid); lo, hi = wilson(ok, len(valid))
    print(f"\n=== {name}: {len(rows)} scenes, {len(rows)-len(valid)} without a valid instruction")
    print(f"success {ok}/{len(valid)} = {100*ok/len(valid):.1f}%  (Wilson 95% {100*lo:.1f}-{100*hi:.1f})")
    print("status:", dict(collections.Counter(r["status"] for r in valid)))
    tch = sum(r["touched"] for r in valid); print(f"touched a non-target fruit: {tch}/{len(valid)} = {100*tch/len(valid):.1f}%")
    st = np.array([r["steps"] for r in valid if r["status"] == "ok"]); print(f"steps per successful episode: mean {st.mean():.0f}, median {np.median(st):.0f}, p90 {np.percentile(st,90):.0f}, max {st.max()}")
    print("by family:", {f: f"{sum(r['status']=='ok' for r in valid if r['family']==f)}/{sum(1 for r in valid if r['family']==f)}" for f in sorted({r['family'] for r in valid})})
    vt = np.array([r["vis_target"] for r in valid]); print(f"target visible fraction: mean {vt.mean():.2f}, <0.5: {(vt<0.5).mean()*100:.0f}%, <0.3: {(vt<0.3).mean()*100:.0f}%")
    for lo_, hi_ in [(0.2, 0.4), (0.4, 0.7), (0.7, 1.01)]:
        sel = [r for r in valid if lo_ <= r["vis_target"] < hi_]
        if sel: print(f"  target vis [{lo_:.1f},{min(hi_,1):.1f}): success {sum(r['status']=='ok' for r in sel)}/{len(sel)}, touched {sum(r['touched'] for r in sel)}")
    ncf = np.array([r["n_cf"] for r in valid]); print(f"scenes/instructions with >=1 valid same-scene counterfactual: {(ncf>0).mean()*100:.0f}% (mean {ncf.mean():.1f})")
    gap = np.array([r["gap_app"] for r in valid]); print(f"planned approach gap: mean {gap.mean()*100:.1f} cm, <0: {(gap<0).mean()*100:.0f}%")
    rolls = collections.Counter(r["roll"] for r in valid); print("chosen rolls:", dict(rolls))
