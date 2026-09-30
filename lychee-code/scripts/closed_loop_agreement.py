"""Closed-loop agreement of selection-track selectors: does the scripted expert detach the fruit the network selected?
   python scripts/closed_loop_agreement.py
For each arm and split with results/eval/sel_<arm>__<split>.jsonl: closed-loop PTA / TSA on the first N scenes, the offline selection PTA / TSA on the SAME scenes, the
share of episodes in which the first detached fruit equals the selected one, and the harvest success given a correct selection.
Writes results/eval/closed_loop_agreement.md and ../tables/closed_loop.tex."""
import glob, json, os, sys

sys.path.insert(0, ".")
from lychee import evalkit

LAB = {"r1_film": "R1", "r0_rho00_film": r"R0, $\rho=0$", "r0_rho90_film": r"R0, $\rho=0.9$"}
SPL = [("iid", "IID"), ("sal", "Sal.-rev."), ("occ", "Occl."), ("dens", "Dens."), ("attr", "Attr.")]


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


arms = sys.argv[1:] or ["r1_film", "r0_rho00_film", "r0_rho90_film"]
lines = ["| arm | split | scenes | closed-loop PTA | offline PTA (same scenes) | closed-loop TSA | offline TSA | first detached = selected | success given correct selection | clean success given correct selection |",
         "|---|---|---|---|---|---|---|---|---|---|"]
tex, stat = [], []
for sp, spname in SPL:
    for arm in arms:
        cl, off = load(f"results/eval/sel_{arm}__{sp}.jsonl"), load(f"results/eval/s_{arm}__{sp}.jsonl")
        if not cl or not off:
            continue
        keys = {r["index"] for r in cl}                                    # the offline sets are rendered from the same scene indices of the same split
        off = [r for r in off if r["index"] in keys]
        sc, so = evalkit.summarize(cl)["all"], evalkit.summarize(off)["all"]
        valid = [r for r in cl if r.get("selected", -2) >= 0]
        agree = sum(r["first_detached"] == r["selected"] for r in valid) / max(1, len(valid))
        good = [r for r in cl if r.get("selected", -2) in r["targets"]]
        succ = sum(r["success"] for r in good) / max(1, len(good))
        clean = sum(bool(r["success"]) and not r.get("touched_nontarget", False) for r in good) / max(1, len(good))     # clean success: no non-target touched
        stat.append((agree, succ, clean, so["PTA_sel"][0] - sc["PTA_sel"][0], sp))
        tex.append(f"{LAB.get(arm, arm)} & {spname} & {sc['n_pairs']} & {evalkit.fmt(sc['PTA_sel']).strip()} & {evalkit.fmt(so['PTA_sel']).strip()} & {100*agree:.1f} & {100*succ:.1f} & {100*clean:.1f}")
        lines.append(f"| {arm} | {sp} | {sc['n_pairs']} | {evalkit.fmt(sc['PTA_sel'])} | {evalkit.fmt(so['PTA_sel'])} | {100*sc['TSA'][0]:.1f} | {100*so['TSA'][0]:.1f} | "
                     f"{100*agree:.1f}% (n={len(valid)}) | {100*succ:.1f}% (n={len(good)}) | {100*clean:.1f}% |")
text = "\n".join(lines)
print(text)
os.makedirs("../tables", exist_ok=True)
if stat:                                                                    # ranges quoted in the text (percent; the gap is offline minus closed-loop PTA in points)
    ag, su, cl_, gp = ([s[i] for s in stat] for i in range(4))
    dens = [s[2] for s in stat if s[4] == "dens"]
    mac = {"clAgreeMin": 100 * min(ag), "clAgreeMax": 100 * max(ag), "clSuccMin": 100 * min(su), "clSuccMax": 100 * max(su),
           "clCleanMin": 100 * min(cl_), "clCleanMax": 100 * max(cl_), "clDirtyMin": 100 - 100 * max(cl_), "clDirtyMax": 100 - 100 * min(cl_), "clGapMax": 100 * max(gp), "clDensClean": 100 * min(dens) if dens else float("nan")}
    open("../tables/closed_macros.tex", "w", newline="\n").write("".join(f"\\providecommand{{\\{k}}}{{{v:.1f}}}\n" for k, v in mac.items()))
open("../tables/closed_loop.tex", "w", newline="\n").write(" \\\\\n".join(tex) + "\n")     # rows are separated by \\ ; main.tex supplies the last one
open("results/eval/closed_loop_agreement.md", "w", encoding="utf-8", newline="\n").write(text + "\n")
