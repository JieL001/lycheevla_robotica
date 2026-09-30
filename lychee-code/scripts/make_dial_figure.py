"""Bias-dial figure: PTA on IID and on the saliency-reversed split against the dial rho for the unpaired arms R0_rho (pixel selectors solid, privileged fruit-table selectors
dashed), on all pairs (top row) and on relational pairs (depth and ordinal commands, bottom row), with the paired arms (R1, R1u) as horizontal reference bands.  Pixel selectors:
mean over the available training seeds, bars = seed range, dots = seeds (Wilson 95% intervals over scenes where an arm has one seed); the abscissa is -log10(1 - rho) so that the strong settings 0.9, 0.97 and 0.99 are not squeezed into the corner, and the tick labels give the
share of training commands that name the most salient fruit (measured on the rendered sets; it saturates near 91 %).
   python scripts/make_dial_figure.py   ->  ../figs/fig_dial.pdf / .png"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lychee import evalkit

OUT = "../figs"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})
RHOS = [(0.0, "00"), (0.5, "50"), (0.9, "90"), (0.97, "97"), (0.99, "99")]
REFS = [("r1_film", "R1 (paired)", "#1b7837"), ("r1u_film", "R1u (independent pair)", "#762a83")]
REL = ("mat_depth", "mat_ordinal")


def load(tag, split):
    path = f"results/eval/s_{tag}__{split}.jsonl"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def pta(tag, split, fams=None):
    r = load(tag, split)
    if not r:
        return None
    if fams is not None:
        r = [x for x in r if x["family"] in fams]
    pt = evalkit.pair_table(r)
    k = sum(1 for v in pt.values() if v["plus"]["target_correct"] and v["minus"]["target_correct"])
    return evalkit.wilson_ci(k, len(pt))


def seed_vals(tag, split, fams=None):
    """PTA (fraction) of every available training seed of an arm (files s_<tag>.jsonl, s_<tag>_s1.jsonl, ...)."""
    out = []
    for suf in ("", "_s1", "_s2", "_s3", "_s4"):
        v = pta(tag + suf, split, fams)
        if v:
            out.append(v[0])
    return out


def xpos(rho):
    return -np.log10(1.0 - rho)


def achieved(tag):
    """Measured share of training commands whose target set contains the most salient fruit (scripts/bias_dial_measured.py)."""
    try:
        m = json.load(open("results/bias_dial_measured.json"))
        return f"{100 * m['rho' + tag]['share_top_in_targets']:.0f}%"
    except Exception:
        return "?"


def curve(ax, fmt, label, color, ls, split, fams, lows, seeds=False):
    """seeds=True: mean over the available training seeds with the seed range as bars (Wilson intervals of seed 0 where an arm has one seed only) and the seeds as dots."""
    xs, ys, lo, hi = [], [], [], []
    for rho, tag in RHOS:
        name = fmt.format(tag)
        sv = seed_vals(name, split, fams) if seeds else []
        v = pta(name, split, fams)
        if len(sv) > 1:
            m = float(np.mean(sv))
            xs.append(xpos(rho)); ys.append(100 * m); lo.append(100 * (m - min(sv))); hi.append(100 * (max(sv) - m)); lows.append(100 * min(sv))
            ax.scatter([xpos(rho)] * len(sv), [100 * s for s in sv], s=6, color=color, alpha=0.45, zorder=2, lw=0)
        elif v:
            xs.append(xpos(rho)); ys.append(100 * v[0]); lo.append(100 * (v[0] - v[1])); hi.append(100 * (v[2] - v[0])); lows.append(100 * v[1])
    if xs:
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker="o", ms=3.5, lw=1.2, ls=ls, capsize=2, label=label)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
    for r, (fams, rowname) in enumerate(((None, "all pairs"), (REL, "depth and ordinal pairs"))):
        lows = []
        for c, (split, title) in enumerate((("iid", "IID scenes"), ("sal", "saliency-reversed scenes"))):
            ax = axes[r, c]
            curve(ax, "r0_rho{}_film", r"R0$_\rho$, pixels", "#b2182b", "-", split, fams, lows, seeds=True)
            curve(ax, "sym_r0_rho{}", r"R0$_\rho$, fruit table", "#e08214", "--", split, fams, lows)
            sv = seed_vals("r0_matched90_film", split, fams)                     # control: the command mix of rho = 0.9 without its correlation with the salient fruit (plotted at rho = 0.9)
            if sv:
                m = float(np.mean(sv))
                ax.errorbar([xpos(0.9)], [100 * m], yerr=[[100 * (m - min(sv))], [100 * (max(sv) - m)]], color="#2166ac", marker="D", ms=4.0, lw=1.0, capsize=2, ls="none",
                            label=r"R0, mix of $\rho=0.9$, no correlation")
                ax.scatter([xpos(0.9)] * len(sv), [100 * s for s in sv], s=6, color="#2166ac", alpha=0.45, zorder=2, lw=0)
                lows.append(100 * min(sv))
            for tag, label, col in REFS:
                sv = seed_vals(tag, split, fams)
                v = pta(tag, split, fams)
                if len(sv) > 1:                                                   # mean and range of the seeds
                    ax.axhline(100 * np.mean(sv), color=col, lw=1.0, ls=":", label=label)
                    ax.axhspan(100 * min(sv), 100 * max(sv), color=col, alpha=0.10, lw=0)
                    lows.append(100 * min(sv))
                elif v:
                    ax.axhline(100 * v[0], color=col, lw=1.0, ls=":", label=label)
                    ax.axhspan(100 * v[1], 100 * v[2], color=col, alpha=0.10, lw=0)
                    lows.append(100 * v[1])
            ax.set_title(f"{title}, {rowname}", fontsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([xpos(x) for x, _ in RHOS]); ax.set_xticklabels([f"{x}\n({achieved(tag)})" for x, tag in RHOS], fontsize=6.5)
            if r == 1:
                ax.set_xlabel(r"bias dial $\rho$ (achieved share)")
        lo = max(0, 5 * np.floor((min(lows) - 3) / 5)) if lows else 0
        for c in range(2):
            axes[r, c].set_ylim(lo, 101)
        axes[r, 0].set_ylabel("PTA (%)")
    axes[0, 0].legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(pad=0.5)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/fig_dial.pdf"); fig.savefig(f"{OUT}/fig_dial.png", dpi=200)
    print(f"wrote {OUT}/fig_dial.pdf")


if __name__ == "__main__":
    main()
