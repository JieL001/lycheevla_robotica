"""PTA on held-out combinations of command family and maturity (attribute-OOD split, 300 pairs per seed), one dot per training seed and a bar at the seed mean:
(a) all pairs, (b) the ordinal pairs, whose held-out combination (ordinal / ripe) is where the seeds differ most.
   python scripts/make_composition_figure.py  ->  ../figs/fig_composition.pdf / .png"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lychee import evalkit

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})
ARMS = [("r0_rho00_film", "R0\n" r"$\rho{=}0$", "#b2182b"), ("r0_rho90_film", "R0\n" r"$\rho{=}0.9$", "0.45"), ("r0p_film", "R0p", "#f4a582"),
        ("r1n_film", "R1n", "#e08214"), ("r1u_film", "R1u", "#762a83"), ("r1_film", "R1", "#1b7837"),
        ("r1_late", "R1\nlate", "#2166ac"), ("r1_late_wide", "late\n0.60M", "#4393c3"), ("r1_token", "R1\ntoken", "#67a9cf"),
        ("mod_free", "modular", "0.2")]
SUFFIX = ("", "_s1", "_s2", "_s3", "_s4")


def seed_values(base, split="attr"):
    """[(overall PTA, ordinal PTA)] in percent, one per available training seed."""
    out = []
    for suf in SUFFIX:
        p = f"results/eval/s_{base}{suf}__{split}.jsonl"
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p)]
        pt = evalkit.pair_table(recs)
        allp = np.mean([bool(v["plus"]["target_correct"] and v["minus"]["target_correct"]) for v in pt.values()])
        fam = evalkit.summarize(recs, by="family")
        out.append((100 * allp, 100 * fam["mat_ordinal"]["PTA_sel"][0]))
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), sharex=True)
    rng = np.random.RandomState(0)
    for j, (ax, title) in enumerate(zip(axes, ("all pairs", "ordinal pairs (held out: ordinal / ripe)"))):
        for i, (base, label, col) in enumerate(ARMS):
            v = seed_values(base)
            if not v:
                continue
            ys = [t[j] for t in v]
            xs = i + (rng.uniform(-0.12, 0.12, len(ys)) if len(ys) > 1 else 0)
            ax.scatter(xs, ys, s=14, color=col, zorder=3, edgecolor="white", linewidth=0.4)
            if len(ys) > 1:
                ax.hlines(np.mean(ys), i - 0.3, i + 0.3, color=col, lw=1.6, zorder=2)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a[1] for a in ARMS], fontsize=6.5)
        ax.set_ylabel("PTA (%)"); ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylim(0, 100); axes[1].set_ylim(0, 100)
    fig.text(0.5, -0.02, "one dot per training seed, bar: seed mean (arms without a bar have one seed)", ha="center", fontsize=7, color="0.3")
    fig.tight_layout()
    os.makedirs("../figs", exist_ok=True)
    fig.savefig("../figs/fig_composition.pdf", bbox_inches="tight"); fig.savefig("../figs/fig_composition.png", dpi=200, bbox_inches="tight")
    print("wrote ../figs/fig_composition.pdf")


if __name__ == "__main__":
    main()
