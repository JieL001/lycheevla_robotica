"""One scripted-expert rollout as a strip of external-view frames (the manipulation that the selection track leaves to the expert): the first frame of each phase.
   python scripts/make_rollout_figure.py [scene index of the IID split]   ->  ../figs/fig_rollout.pdf / .png   (needs the simulator; about 1 minute)"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lychee import evalkit
from lychee.expert import Expert
from lychee.splits import sample_config

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3
cfg = sample_config("iid", idx)
env = evalkit.make_env(obs_mode="rgb", img=256, hand_img=64, render_mode="rgb_array")
u = env.unwrapped
env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which="plus"))
ex = Expert(env, target=u.target_idx)
frames, phases, seen = [], [], []
for step in range(320):
    a = ex.act()
    env.step(a)
    if ex.phase not in seen:
        seen.append(ex.phase)
        img = env.render()
        img = img[0].cpu().numpy() if hasattr(img, "cpu") else img
        frames.append(np.asarray(img)); phases.append((ex.phase, step))
    if ex.done:
        break
print("phases", phases, "detach order", u.detach_order, "target", u.target_idx)
n = len(frames)
fig, axes = plt.subplots(1, n, figsize=(1.55 * n, 1.9))
for ax, im, (ph, st) in zip(np.atleast_1d(axes), frames, phases):
    h, w = im.shape[:2]
    ax.imshow(im[int(0.12 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)]); ax.axis("off")
    ax.set_title(f"{ph}\n(step {st})", fontsize=6.5, pad=2)
fig.suptitle(f"“{u.instruction}”", fontsize=7, y=0.02, va="bottom")
fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
os.makedirs("../figs", exist_ok=True)
fig.savefig("../figs/fig_rollout.pdf"); fig.savefig("../figs/fig_rollout.png", dpi=200)
print("wrote ../figs/fig_rollout.pdf")
