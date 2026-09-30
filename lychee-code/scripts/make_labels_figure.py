"""PTA against the number of distinct (scene, target set) training labels: data-scaling curves of R0_0 (unpaired) and R1 (paired, whole-scene subsets), and the
single points of the controls R1u (independent commands, 6,240 distinct labels) and R1n (paraphrase, 4,000) and of the biased R0_0.9 (8,000).  Seed 0; Wilson 95% intervals.
   python scripts/make_labels_figure.py   ->  ../figs/fig_labels.pdf / .png  and  ../tables/labels_macros.tex"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from lychee import evalkit

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})
LAB = json.load(open("results/bias_dial_measured.json"))
WORD = {1000: "OneK", 2000: "TwoK", 4000: "FourK", 8000: "EightK"}


def pta(tag, split):
    p = f"results/eval/s_{tag}__{split}.jsonl"
    if not os.path.exists(p):
        return None
    r = [json.loads(l) for l in open(p)]
    pt = evalkit.pair_table(r)
    k = sum(1 for v in pt.values() if v["plus"]["target_correct"] and v["minus"]["target_correct"])
    return evalkit.wilson_ci(k, len(pt))


def series(base, budgets, full_tag):
    xs, ys = [], []
    for n in budgets:
        xs.append(n); ys.append((f"{base}_n{n}",))
    xs.append(8000); ys.append((full_tag,))
    return xs, [y[0] for y in ys]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    macros = [f"\\providecommand{{\\lb{a}{w}{s}}}{{??}}" for a in ("Zero", "One") for w in WORD.values() for s in ("Iid", "Sal")]
    for ax, (split, title) in zip(axes, (("iid", "IID scenes"), ("sal", "saliency-reversed scenes"))):
        for (base, full, label, col, mk) in (("r0_rho00_film", "r0_rho00_film", r"R0$_{\rho=0}$ (one command per scene)", "#b2182b", "o"),
                                             ("r1_film", "r1_film", "R1 (both commands of a pair)", "#1b7837", "s")):
            xs, tags = series(base, (1000, 2000, 4000), full)
            pts = [(x, pta(t, split)) for x, t in zip(xs, tags)]
            pts = [(x, v) for x, v in pts if v]
            if pts:
                ax.errorbar([x for x, _ in pts], [100 * v[0] for _, v in pts], yerr=[[100 * (v[0] - v[1]) for _, v in pts], [100 * (v[2] - v[0]) for _, v in pts]],
                            color=col, marker=mk, ms=3.5, lw=1.1, capsize=2, label=label)
                for x, v in pts:
                    nm = f"\\lb{'Zero' if base.startswith('r0') else 'One'}{WORD[x]}{split.capitalize()}"      # macro names are letters only
                    macros.append(f"\\providecommand{{{nm}}}{{??}}\\renewcommand{{{nm}}}{{{100*v[0]:.1f}}}")
        for tag, x_key, label, col, mk in (("r1u_film", "indep", "R1u (independent commands)", "#762a83", "D"), ("r1n_film", "same", "R1n (paraphrase)", "#e08214", "^"),
                                           ("r0p_film", "r0p", "R0p (first command of a pair)", "#f4a582", "v"),
                                           ("r0_rho90_film", "rho90", r"R0$_{\rho=0.9}$ (biased)", "0.45", "o")):
            v = pta(tag, split)
            if v and (x_key in LAB or x_key == "r0p"):
                x = 4000 if x_key == "r0p" else LAB[x_key]["distinct_labels"]      # R0p: one command on each of the 4,000 scenes of R1
                x = x * {"same": 0.93, "r0p": 1.07}.get(x_key, 1.0)                # the two 4,000-label controls are drawn side by side (the label count is the same)
                ax.errorbar([x], [100 * v[0]], yerr=[[100 * (v[0] - v[1])], [100 * (v[2] - v[0])]], color=col, marker=mk, ms=4.5, lw=0, elinewidth=1.0, capsize=2,
                            mfc="white" if x_key == "rho90" else col, label=label)
        ax.set_xscale("log"); ax.set_xticks([1000, 2000, 4000, 8000]); ax.set_xticklabels(["1k", "2k", "4k", "8k"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("distinct (scene, target set) training labels"); ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("PTA (%)"); axes[0].legend(frameon=False, fontsize=6.0, loc="lower right")
    fig.tight_layout(pad=0.5)
    os.makedirs("../figs", exist_ok=True)
    fig.savefig("../figs/fig_labels.pdf"); fig.savefig("../figs/fig_labels.png", dpi=200)
    open("../tables/labels_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    print("wrote ../figs/fig_labels.pdf")


if __name__ == "__main__":
    main()
