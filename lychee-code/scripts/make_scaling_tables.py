"""Data-scaling and scene-binding tables of the selection track.
   python scripts/make_scaling_tables.py   ->  ../tables/scaling.tex, ../tables/binding.tex, ../tables/scaling_macros.tex, results/eval/scaling.md

Scaling: PTA (Wilson 95 % intervals over the 600 scenes) of R0_0 and R1 trained on 1,000 / 2,000 / 4,000 / 8,000 labelled samples (seed 0).
Binding: an arm is trained on the first K scenes of the paired set (5 % of them held out for validation, exactly as train_select.py does) for a
fixed number of updates, either with one command per scene (bind_r0: the first command only) or with both commands (bind_r1); it is then tested
on its own TRAINING scenes with both commands (split cf).  On the second command of bind_r0, which the network never saw on that scene, an answer is
faithful if the selected fruit is a target of the new command and biased if it is a target of the command the scene was trained with (the outcome
split of LIBERO-CF).  The last columns give PTA on new IID scenes."""
import json, os, sys

import numpy as np

sys.path.insert(0, ".")
from lychee import evalkit

from lychee.paths import DATA as _DATA

E = "results/eval"
DATA = f"{_DATA}/select/paired"


def load(name):
    p = f"{E}/{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def pta(name):
    r = load(name)
    if not r:
        return None
    pt = evalkit.pair_table(r)
    k = sum(1 for v in pt.values() if v["plus"]["target_correct"] and v["minus"]["target_correct"])
    return evalkit.wilson_ci(k, len(pt))


def cell(v):
    return "--" if v is None else f"{100*v[0]:.1f} [{100*v[1]:.0f}, {100*v[2]:.0f}]"


def train_scenes(K, seed=0, val_frac=0.05):
    """The scene indices train_select.py trains on for --scenes K (and those it holds out)."""
    if os.path.exists(f"{DATA}/meta.npz"):
        ok = np.load(f"{DATA}/meta.npz")["ok"]
        assert ok.all(), "the fallback below assumes that every scene of the paired set is valid"
    else:
        ok = np.ones(4000, bool)                             # rendered set not on disk: all 4,000 scenes of the paired set are valid (checked whenever the set is present)
    scenes = np.flatnonzero(ok)[:K]
    perm = np.random.RandomState(seed).permutation(scenes)
    n_val = max(20, int(len(scenes) * val_frac))
    return set(int(i) for i in perm[n_val:]), set(int(i) for i in perm[:n_val])


def main():
    md, macros = [], []
    # ---- data scaling
    rows = []
    names = {1000: "One", 2000: "Two", 4000: "Four", 8000: "Eight"}
    for label, base, akey in ((r"R0$_{\rho=0}$", "r0_rho00_film", "Rzero"), ("R1", "r1_film", "Rone")):
        for split, sname, skey in (("iid", "IID", "Iid"), ("sal", "saliency-reversed", "Sal")):
            vs = [pta(f"s_{base}_n{n}__{split}") for n in (1000, 2000, 4000)] + [pta(f"s_{base}__{split}")]
            cells = [cell(v) for v in vs]
            rows.append(f"{label}, {sname} & " + " & ".join(cells))
            md.append(f"| {label} {sname} | " + " | ".join(cells) + " |")
            for n, v in zip((1000, 2000, 4000, 8000), vs):
                if v is not None:
                    macros.append(f"\\providecommand{{\\sc{akey}{names[n]}{skey}}}{{??}}\\renewcommand{{\\sc{akey}{names[n]}{skey}}}{{{100*v[0]:.1f}}}")
    os.makedirs("../tables", exist_ok=True)
    open("../tables/scaling.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    print("\n".join(["| arm | 1k | 2k | 4k | 8k |", "|---|---|---|---|---|"] + md))
    # ---- scene binding
    brow, bmd = [], ["", "| arm | K | train scenes | 1st cmd correct | 2nd cmd faithful | 2nd cmd biased | 2nd other | PTA train scenes | PTA new scenes |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, label, seen in (("bind_r0", "one command per scene (R0)", "1"), ("bind_r1", "both commands (R1)", "2")):
        for K in (100, 1000):
            recs = load(f"s_{arm}_K{K}__cf")
            if not recs:
                continue
            tr, va = train_scenes(K)
            by = {}
            for r in recs:
                by.setdefault(r["index"], {})[r["which"]] = r
            n = f = b = o = c1 = pair_ok = 0
            ch_f = ch_b = 0.0                                              # random-choice level of the faithful and the biased answer
            for i, v in by.items():
                if i not in tr or len(v) < 2:
                    continue
                p, m = v["plus"], v["minus"]
                sel = m["first_detached"]
                n += 1
                ch_f += len(m["targets"]) / m["n_fruit"]; ch_b += len(p["targets"]) / p["n_fruit"]
                c1 += bool(p["target_correct"])
                pair_ok += bool(p["target_correct"]) and bool(m["target_correct"])
                if sel in m["targets"]:
                    f += 1
                elif sel in p["targets"]:
                    b += 1
                else:
                    o += 1
            if n == 0:
                continue
            new = pta(f"s_{arm}_K{K}__iid")
            brow.append(f"{label} & {K} & {len(tr)} & {100*c1/n:.0f} & {100*f/n:.0f} & {100*b/n:.0f} & {100*o/n:.0f} & {100*pair_ok/n:.0f} & {cell(new).split(' [')[0] if new else '--'}")
            bmd.append(f"| {arm} | {K} | {len(tr)} | {100*c1/n:.1f} | {100*f/n:.1f} | {100*b/n:.1f} | {100*o/n:.1f} | {100*pair_ok/n:.1f} | {cell(new)} |")
            tag = ("Bz" if arm == "bind_r0" else "Bo") + ("Hundred" if K == 100 else "Thousand")
            for nm, val in (("First", c1), ("Faith", f), ("Biased", b), ("Other", o), ("Pta", pair_ok)):
                macros.append(f"\\providecommand{{\\{tag}{nm}}}{{??}}\\renewcommand{{\\{tag}{nm}}}{{{100*val/n:.0f}}}")
            if new:
                macros.append(f"\\providecommand{{\\{tag}New}}{{??}}\\renewcommand{{\\{tag}New}}{{{100*new[0]:.0f}}}")
            macros.append(f"\\providecommand{{\\{tag}ChanceFaith}}{{??}}\\renewcommand{{\\{tag}ChanceFaith}}{{{100*ch_f/n:.0f}}}"
                          f"\\providecommand{{\\{tag}ChanceBiased}}{{??}}\\renewcommand{{\\{tag}ChanceBiased}}{{{100*ch_b/n:.0f}}}")
    open("../tables/binding.tex", "w", newline="\n").write((" \\\\\n".join(brow) + "\n") if brow else "")
    open("../tables/scaling_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    print("\n".join(bmd))
    open("results/eval/scaling.md", "w", encoding="utf-8", newline="\n").write("\n".join(md + bmd) + "\n")


if __name__ == "__main__":
    main()
