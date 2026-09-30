"""Figures for the manuscript (all generated from the code in this folder).  usage: python scripts/make_figures.py"""
import json, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "../figs"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})


def fig_pair(env, u, cfgs, name):
    from lychee.layout import DEFAULT_CAM, R_FRUIT
    f_px = 128 / np.tan(DEFAULT_CAM.fov / 2)
    fig, axes = plt.subplots(len(cfgs), 2, figsize=(6.6, 2.9 * len(cfgs)))
    axes = np.atleast_2d(axes)
    for row, cfg in zip(axes, cfgs):
        for ax, which, col in zip(row, ("plus", "minus"), ("#1a9850", "#d6008a")):
            obs, _ = env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which=which))
            img = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
            ep = cfg.plus if which == "plus" else cfg.minus
            ax.imshow(img[60:170, :], extent=(0, 256, 170, 60))
            for t in ep.targets:
                uu, dd, vv = DEFAULT_CAM.project(u.rest[t])
                r = f_px * np.tan(np.arcsin(R_FRUIT / np.linalg.norm(u.rest[t] - DEFAULT_CAM.eye)))
                ax.add_patch(Circle((uu * 256, vv * 256), r + 4, fill=False, ec=col, lw=1.6))
            ax.set_title(f"\u201c{ep.text}\u201d", fontsize=7.5, color=col, pad=3)
            ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def fig_gallery(env, u, name):
    from lychee.splits import sample_config
    labels = [("iid", "IID"), ("occ_ood", "Occlusion-OOD"), ("density_ood", "Density-OOD")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.95))
    for ax, (split, lab) in zip(axes, labels):
        cfg = sample_config(split, 3)
        obs, _ = env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which="plus"))
        img = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        ax.imshow(img[60:170, :]); ax.set_title(f"{lab}  (n={u.n_fruit}, target vis {min(u.vis[t] for t in cfg.plus.targets):.2f})", fontsize=7, pad=2)
        ax.axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(fig)


def fig_visibility():
    d = np.load("results/visibility_check.npz")
    a, r = d["analytic"], d["rendered"]
    fig, ax = plt.subplots(figsize=(2.9, 2.7))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.scatter(a, r, s=6, alpha=0.6, color="#2166ac", linewidths=0)
    ax.set_xlabel("analytic visibility (ray casting)"); ax.set_ylabel("rendered visibility (segmentation)")
    ax.set_title(f"n={len(a)} fruits, Pearson r={np.corrcoef(a, r)[0,1]:.3f}, MAE={np.abs(a-r).mean():.3f}", fontsize=7)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_visibility.pdf"); fig.savefig(f"{OUT}/fig_visibility.png", dpi=200); plt.close(fig)


def fig_feasibility():
    d = json.load(open("results/family_feasibility.json"))
    fams = ["mat_any", "mat_side", "mat_ordinal", "mat_depth", "mat"]
    nice = {"mat_any": "any-of", "mat_side": "side", "mat_ordinal": "ordinal", "mat_depth": "depth", "mat": "unique maturity"}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    w = 0.26
    for k, (lab, col) in enumerate(zip(d.keys(), ("#a6cee3", "#1f78b4", "#08306b"))):
        ax.bar(np.arange(len(fams)) + (k - 1) * w, [d[lab][f] for f in fams], w, label=lab.replace("N=", "N = "), color=col)
    ax.set_xticks(np.arange(len(fams))); ax.set_xticklabels([nice[f] for f in fams], rotation=20, ha="right")
    ax.set_ylabel("scenes admitting a pair"); ax.set_ylim(0, 1.0); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4); fig.savefig(f"{OUT}/fig_feasibility.pdf"); fig.savefig(f"{OUT}/fig_feasibility.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    from lychee import evalkit
    from lychee.splits import sample_config
    env = evalkit.make_env(obs_mode="rgb", img=256)
    u = env.unwrapped
    cfgs = []
    for idx in range(0, 40):
        c = sample_config("iid", idx)
        if c.family in ("mat_any", "mat_side") and c.plus.text and len(cfgs) < 2 and (not cfgs or c.family != cfgs[0].family):
            cfgs.append(c)
    fig_pair(env, u, cfgs, "fig_pair")
    fig_gallery(env, u, "fig_gallery")
    fig_visibility(); fig_feasibility()
    print("figures written to", OUT)
