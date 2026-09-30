"""Total compute of the study, read from the logs (so that the paper quotes measured numbers).
   python scripts/compute_summary.py   ->  results/compute_summary.json and ../tables/compute_macros.tex
Training: the last "(NNNs)" of every results/select_train_*.log (wall-clock seconds on the shared laptop GPU) and results/train_*.log of the
end-to-end track; rendering: the "seconds" of every info.json of <data>/select*; demonstrations: the shard manifests if present.
The logs and the rendered data are not part of the repository: without them the shipped results/compute_summary.json is used, so that the macro file regenerates."""
import glob, json, os, re, sys

sys.path.insert(0, ".")
from lychee.paths import DATA

JS = "results/compute_summary.json"
res = dict()
sel = {}
for p in sorted(glob.glob("results/select_train_*.log")):
    txt = open(p, errors="ignore").read()
    m = re.findall(r"epoch (\d+) loss [\d.]+ val selection accuracy [\d.]+ \((\d+)s\)", txt)
    if m and "saved" in txt:
        sel[os.path.basename(p)[len("select_train_"):-4]] = int(m[-1][1])
e2e = {}
for p in sorted(glob.glob("results/train_*.log")):
    txt = open(p, errors="ignore").read()
    m = re.findall(r"\((\d+)s\)", txt)
    if m:
        e2e[os.path.basename(p)[len("train_"):-4]] = int(m[-1])
render = {}
for p in sorted(glob.glob(f"{DATA}/select*/*/info.json")):
    try:
        d = json.load(open(p))
        if d.get("reused_images_of"):                              # labels only (scripts/make_matched_dataset.py): the images are those of another set
            continue
        render[os.path.relpath(os.path.dirname(p), DATA).replace("\\", "/")] = (d.get("seconds", 0), d.get("ok", 0))
    except Exception:
        pass
if sel or e2e or render:                                           # measured from the logs and the rendered data
    res["select_arms_trained"] = len(sel)
    res["select_train_hours"] = round(sum(sel.values()) / 3600, 2)
    res["select_train_minutes_range"] = [round(min(sel.values()) / 60, 1), round(max(sel.values()) / 60, 1)] if sel else None
    res["select_train_minutes_sorted"] = sorted(round(v / 60, 3) for v in sel.values())
    res["e2e_train_hours"] = round(sum(e2e.values()) / 3600, 2)
    res["render_sets"] = len(render)
    res["render_hours"] = round(sum(s for s, _ in render.values()) / 3600, 2)
    res["render_scenes"] = int(sum(n for _, n in render.values()))
    demo_s = 0.0
    for p in glob.glob(f"{DATA}/*/shard_manifest.json"):
        try:
            d = json.load(open(p))
            demo_s += float(d.get("seconds", 0) or 0)
        except Exception:
            pass
    res["demo_hours_reported"] = round(demo_s / 3600, 2)
    json.dump(res, open(JS, "w"), indent=1)
    print("largest training runs:", sorted(sel.items(), key=lambda kv: -kv[1])[:3])
else:
    res = json.load(open(JS))                                       # no logs, no data: the shipped summary
print(json.dumps(res, indent=1))
macros = [f"\\providecommand{{\\cmpArms}}{{{res['select_arms_trained']}}}", f"\\providecommand{{\\cmpTrainHours}}{{{res['select_train_hours']:.0f}}}",
          f"\\providecommand{{\\cmpRenderHours}}{{{res['render_hours']:.1f}}}", f"\\providecommand{{\\cmpScenes}}{{{res['render_scenes']:,}}}".replace(",", "{,}")]
mins = res.get("select_train_minutes_sorted") or []
if mins:  # per-arm wall-clock range on the shared laptop (median as the typical value)
    macros += [f"\\providecommand{{\\cmpMinLo}}{{{mins[0]:.0f}}}", f"\\providecommand{{\\cmpMinHi}}{{{mins[-1]:.0f}}}",
               f"\\providecommand{{\\cmpMinMed}}{{{mins[len(mins) // 2]:.0f}}}"]
os.makedirs("../tables", exist_ok=True)
open("../tables/compute_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
