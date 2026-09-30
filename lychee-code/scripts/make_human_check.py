"""Build a small human-validation packet: are the commands' target sets unambiguous to a human viewer?
   python scripts/make_human_check.py --n 40       # writes results/human_check/{items.pdf, items/*.png, answers_template.csv, key.json}
   python scripts/score_human_check.py answers.csv # after annotators fill the sheet
Each item shows the third-person view (fruit numbered) and a command; the annotator writes the number of the
fruit they would pick (any acceptable one for 'any' commands), or 'A' if the command is ambiguous / no fruit fits.
"""
import argparse, csv, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lychee import evalkit
from lychee.layout import R_FRUIT
from lychee.splits import sample_config

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="results/human_check")
a = ap.parse_args()
os.makedirs(f"{a.out}/items", exist_ok=True)
env = evalkit.make_env(obs_mode="rgb", img=256)
u = env.unwrapped
mix = ["iid"] * 4 + ["occ_ood"] * 2 + ["density_ood"] * 2 + ["attr_ood"] * 1 + ["saliency_rev"] * 1
items, key = [], []
k = 0
idx = 500                                          # index range disjoint from every split's evaluation range
while len(items) < a.n:
    split = mix[len(items) % len(mix)]
    cfg = sample_config(split, idx); idx += 1
    which = "plus" if len(items) % 2 == 0 else "minus"
    ep = cfg.plus if which == "plus" else cfg.minus
    obs, _ = env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which=which))
    items.append((split, cfg, which, ep, obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy().copy(),
                  [u.layout.cam.project(p) for p in u.rest], u.layout.cam, [int(m) for m in u.mat_of]))
with PdfPages(f"{a.out}/items.pdf") as pdf:
    for page in range(0, len(items), 4):
        fig, axes = plt.subplots(4, 1, figsize=(8.0, 11.0))
        for ax, (i, it) in zip(axes, enumerate(items[page:page + 4], start=page)):
            split, cfg, which, ep, img, proj, cam, mats = it
            ax.imshow(img[40:200], extent=(0, 256, 200, 40))
            f_px = 128 / np.tan(cam.fov / 2)
            for j, (uu, dd, vv) in enumerate(proj):
                ax.text(uu * 256, vv * 256 - 16, str(j), color="white", fontsize=9, ha="center", va="center", weight="bold",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="black", ec="none", alpha=0.75))
            ax.set_title(f"Item {i + 1}:  “{ep.text}”", fontsize=12, loc="left")
            ax.axis("off")
            key.append(dict(item=i + 1, split=split, index=cfg.index, which=which, text=ep.text, family=ep.spec.family,
                            targets=list(ep.targets), n_fruit=len(proj)))
            plt.imsave(f"{a.out}/items/item_{i+1:02d}.png", img[40:200])
        fig.suptitle("Write the NUMBER of the fruit you would pick (any acceptable one for 'any' commands);\nwrite A if the command is ambiguous or no fruit fits.", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); pdf.savefig(fig); plt.close(fig)
json.dump(key, open(f"{a.out}/key.json", "w"), indent=1)
with open(f"{a.out}/answers_template.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["item", "annotator", "answer"])
    for kk in key:
        w.writerow([kk["item"], "", ""])
print(f"wrote {len(items)} items to {a.out}")
