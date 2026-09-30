"""Analytic visibility versus rendered visibility, as the error per bin of analytic visibility (no simulator needed; reads results/visibility_check_off.npz
written by scripts/check_visibility.py).  Bars: mean absolute difference; whiskers: 90th percentile; numbers under the axis: fruit per bin.  Also prints the
error in the band around the training / occlusion boundary (0.3-0.5) and writes ../tables/visibility_macros.tex.
   python scripts/make_visibility_figure.py   ->  ../figs/fig_visibility.pdf / .png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})
d = np.load("results/visibility_check_off.npz")
a, r = d["analytic"], d["rendered"]
scenes = int(d["scenes"]) if "scenes" in d.files else 0
err = np.abs(a - r)
edges = np.linspace(0, 1, 11)
idx = np.clip(np.digitize(a, edges) - 1, 0, 9)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.axvspan(0.3, 0.5, color="#fde0c5", lw=0, zorder=0)
labels = []
for k in range(10):
    m = idx == k
    c = 0.5 * (edges[k] + edges[k + 1])
    labels.append(f"{c:.2f}\n{int(m.sum())}")
    if m.any():
        ax.bar(c, err[m].mean(), width=0.085, color="#2166ac", zorder=2)
        ax.plot([c, c], [err[m].mean(), np.quantile(err[m], 0.9)], color="#08306b", lw=0.9, zorder=3)
ax.set_xticks(0.5 * (edges[:-1] + edges[1:])); ax.set_xticklabels(labels, fontsize=5.8)
ax.text(0.4, ax.get_ylim()[1] * 0.97, "training / occlusion\nboundary", ha="center", va="top", fontsize=5.5, color="#a65600")
ax.set_xlabel("analytic visibility (bin centre; fruit per bin below)"); ax.set_ylabel("|analytic $-$ rendered|")
ax.set_xlim(0, 1); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.4)
os.makedirs("../figs", exist_ok=True)
fig.savefig("../figs/fig_visibility.pdf"); fig.savefig("../figs/fig_visibility.png", dpi=200)
band = (a >= 0.3) & (a < 0.5)
res = dict(n=len(a), mae=float(err.mean()), median=float(np.median(err)), p90=float(np.quantile(err, 0.9)), pearson=float(np.corrcoef(a, r)[0, 1]),
           band_n=int(band.sum()), band_mae=float(err[band].mean()), band_p90=float(np.quantile(err[band], 0.9)), band_signed=float((r - a)[band].mean()))
print(res)
os.makedirs("../tables", exist_ok=True)
open("../tables/visibility_macros.tex", "w", newline="\n").write("\n".join(
    f"\\providecommand{{\\vis{k}}}{{??}}\\renewcommand{{\\vis{k}}}{{{v}}}" for k, v in
    (("N", res["n"]), ("Scenes", scenes), ("Mae", f"{res['mae']:.3f}"), ("Median", f"{res['median']:.3f}"), ("Pnine", f"{res['p90']:.3f}"), ("Pearson", f"{res['pearson']:.3f}"),
     ("BandN", res["band_n"]), ("BandMae", f"{res['band_mae']:.3f}"), ("BandPnine", f"{res['band_p90']:.3f}"), ("BandSigned", f"{res['band_signed']:+.3f}"))) + "\n")
