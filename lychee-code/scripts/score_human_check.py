"""Score the human validation of the command labels.
   python scripts/score_human_check.py answers_a.csv [answers_b.csv ...] [--key results/human_check_web/key.json]
CSV columns (as copied from the web page): annotator,item,answer,seconds,timestamp; answer = a fruit number or A (ambiguous / no fruit fits).
Reports, with Wilson 95% intervals: the share of answers inside the target set (human target accuracy, overall / per family / per
split / per target-visibility band), the share marked ambiguous, the share of non-target fruit, per-item majority vote and pairwise
agreement between annotators, and the items on which annotators disagree with the key."""
import argparse, csv, itertools, json, sys
import numpy as np

sys.path.insert(0, ".")
from lychee.evalkit import wilson_ci

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--key", default="results/human_check_web/key.json")
a = ap.parse_args()
key = {k["item"]: k for k in json.load(open(a.key))}
rows = []
for path in a.files:
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r.get("answer", "").strip():
            rows.append(dict(item=int(r["item"]), ann=(r.get("annotator", "").strip() or path), ans=r["answer"].strip().upper(),
                             sec=float(r["seconds"]) if r.get("seconds") else float("nan")))
if not rows:
    sys.exit("no answers found")
good = lambda r: r["ans"] != "A" and int(r["ans"]) in key[r["item"]]["targets"]
for r in rows:
    r["good"], r["amb"] = good(r), r["ans"] == "A"
    r["wrong"] = not r["good"] and not r["amb"]


def line(name, rs):
    k = sum(r["good"] for r in rs); n = len(rs)
    p, lo, hi = wilson_ci(k, n)
    return f"{name:22s} n={n:4d}  in target set {100*p:5.1f}% [{100*lo:.0f}, {100*hi:.0f}]  ambiguous {100*np.mean([r['amb'] for r in rs]):4.1f}%  other fruit {100*np.mean([r['wrong'] for r in rs]):4.1f}%"


print(f"{len(rows)} answers from {len({r['ann'] for r in rows})} annotator(s) on {len({r['item'] for r in rows})} items; median {np.nanmedian([r['sec'] for r in rows]):.1f} s per item")
print(line("all", rows))
for name, fn in (("family", lambda r: key[r["item"]]["family"]), ("split", lambda r: key[r["item"]]["split"]),
                 ("target visibility", lambda r: "v<0.4" if key[r["item"]]["target_vis"] < 0.4 else ("0.4-0.7" if key[r["item"]]["target_vis"] < 0.7 else "v>=0.7")),
                 ("fruit count", lambda r: "3-6" if key[r["item"]]["n_fruit"] <= 6 else ("7-10" if key[r["item"]]["n_fruit"] <= 10 else "11+"))):
    print(f"-- by {name}")
    for g in sorted({fn(r) for r in rows}):
        print("  " + line(g, [r for r in rows if fn(r) == g]))
by_item = {}
for r in rows:
    by_item.setdefault(r["item"], []).append(r)
multi = {i: rs for i, rs in by_item.items() if len(rs) >= 2}
if multi:
    agree = [np.mean([x["ans"] == y["ans"] for x, y in itertools.combinations(rs, 2)]) for rs in multi.values()]
    print(f"pairwise agreement between annotators (exact same answer) on {len(multi)} items with >= 2 annotators: {100*np.mean(agree):.1f}%")
    maj = []
    for i, rs in multi.items():
        vals, counts = np.unique([x["ans"] for x in rs], return_counts=True)
        top = vals[counts.argmax()]
        maj.append(top != "A" and int(top) in key[i]["targets"])
    print(f"majority answer inside the target set: {100*np.mean(maj):.1f}% of {len(maj)} items")
bad = sorted(i for i, rs in by_item.items() if any(not r["good"] for r in rs))
print(f"items where at least one annotator disagreed with the key ({len(bad)}):", bad[:60], "..." if len(bad) > 60 else "")
