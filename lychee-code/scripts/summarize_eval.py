import json, sys, glob
sys.path.insert(0, ".")
from lychee import evalkit
split = sys.argv[1] if len(sys.argv) > 1 else "iid"
order = ["expert", "attr_visible", "random", "blind_visible", "blind_front"]
rows, fam_rows, out = [], [], {}
for pol in order:
    try:
        recs = [json.loads(l) for l in open(f"results/eval/{pol}_{split}.jsonl")]
    except FileNotFoundError:
        continue
    s = evalkit.summarize(recs)["all"]; out[pol] = s
    rows.append(f"| {pol} | {s['n_pairs']} | {evalkit.fmt(s['PTA_sel'])} | {evalkit.fmt(s['TSA'])} | {evalkit.fmt(s['success'])} | {evalkit.fmt(s['wrong_target'])} | {evalkit.fmt(s['touched'])} |")
    for fam, m in evalkit.summarize(recs, by="family").items():
        out.setdefault(pol + "_by_family", {})[fam] = m
print(f"### Reference policies on `{split}` (percent, bootstrap 95% CI over pairs)")
print("| policy | pairs | PTA | TSA | success | wrong-target | touched non-target |\n|---|---|---|---|---|---|---|")
print("\n".join(rows))
print("\nPTA by command family (pairs):")
fams = ["mat_any", "mat_side", "mat_ordinal", "mat_depth", "mat"]
print("| policy | " + " | ".join(fams) + " |\n|---|" + "---|" * len(fams))
for pol in order:
    d = out.get(pol + "_by_family")
    if d: print(f"| {pol} | " + " | ".join((f"{100*d[f]['PTA_sel'][0]:.0f} (n={d[f]['n_pairs']})" if f in d else "-") for f in fams) + " |")
json.dump(out, open(f"results/eval/summary_{split}.json", "w"), indent=1, default=float)
