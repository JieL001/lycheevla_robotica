"""Schematic of the selection track: scene image + command -> pointing network -> pointed pixel -> nearest fruit centre -> scripted expert -> PTA over the pair.
   python scripts/make_pipeline_figure.py   ->  ../figs/fig_pipeline.pdf / .png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5})
INK, BOX, ACC, GREY = "#222222", "#eef2f7", "#b2182b", "#666666"


def box(ax, x, y, w, h, text, fc=BOX, ec=INK, lw=0.8, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 15.0); ax.set_ylim(0, 4.8); ax.axis("off")
    # inputs
    box(ax, 0.1, 3.05, 2.7, 1.2, "scene image\n$160\\times256$ px")
    box(ax, 0.1, 0.85, 2.7, 1.5, "command (43 words)\n“pick the ripe lychee\non the left”")
    # network
    box(ax, 3.4, 0.75, 3.7, 3.65, "", fc="#fdf3e7")
    ax.text(5.25, 4.12, "selection network (0.6 M)", ha="center", va="center", fontsize=6.6, fontweight="bold")
    box(ax, 3.6, 2.85, 3.3, 0.8, "conv stem, strides 2, 2, 2, 1\n$\\rightarrow$ 640 tokens (20$\\times$32)", fs=6.3)
    box(ax, 3.6, 1.95, 3.3, 0.6, "2 transformer layers", fs=6.3)
    box(ax, 3.6, 0.95, 3.3, 0.75, "pointing head $\\rightarrow$ 640 logits", fs=6.3)
    arrow(ax, 5.25, 2.85, 5.25, 2.55); arrow(ax, 5.25, 1.95, 5.25, 1.7)
    ax.text(5.25, 0.36, "the command enters by FiLM (default),\nlate fusion or as a token", ha="center", va="center", fontsize=6.0, color=ACC, linespacing=1.25)
    arrow(ax, 2.8, 3.65, 3.6, 3.3)
    arrow(ax, 2.8, 1.6, 3.6, 1.35, color=ACC)
    arrow(ax, 2.8, 2.05, 3.6, 3.05, color=ACC, ls="--", rad=-0.2)
    # decode and execute
    box(ax, 7.9, 2.55, 3.0, 1.65, "arg-max cell $\\rightarrow$\nnearest fruit centre\nwithin 20 px\n(else: wrong)", fs=6.3)
    box(ax, 7.9, 0.75, 3.0, 1.3, "scripted expert\npicks that fruit\n(99% success)", fs=6.3)
    arrow(ax, 7.1, 2.1, 7.9, 3.2); arrow(ax, 9.4, 2.55, 9.4, 2.05)
    # metric
    box(ax, 11.7, 1.5, 3.2, 2.2, "", fc="#eaf3ea")
    ax.text(13.3, 3.38, "paired target accuracy", ha="center", va="center", fontsize=6.6, fontweight="bold")
    ax.text(13.3, 2.45, "a pair counts only if\ncommand $\\ell^+$ selects a fruit in $T^+$\nand $\\ell^-$ selects a fruit in $T^-$\n($T^+\\cap T^-=\\emptyset$)", ha="center", va="center", fontsize=6.0, linespacing=1.3)
    arrow(ax, 10.9, 3.4, 11.7, 3.0); arrow(ax, 10.9, 1.4, 11.7, 2.0)
    ax.text(13.3, 0.85, "command-blind floor: 0 (deterministic),\nat most 1/4 (independent draws)", ha="center", va="center", fontsize=6.0, color=GREY, linespacing=1.3)
    os.makedirs("../figs", exist_ok=True)
    fig.savefig("../figs/fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig("../figs/fig_pipeline.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    print("wrote ../figs/fig_pipeline.pdf")


if __name__ == "__main__":
    main()
