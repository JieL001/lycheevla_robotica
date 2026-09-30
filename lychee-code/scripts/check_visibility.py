"""Analytic ray-cast visibility (layout.fruit_visibility) vs rendered segmentation pixel counts."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from lychee import evalkit
from lychee.layout import DEFAULT_CAM, R_FRUIT
from dataclasses import replace
from lychee.splits import sample_config, SPLITS
DR = sys.argv[2] if len(sys.argv) > 2 else 'off'

env = evalkit.make_env(obs_mode="rgb+segmentation", img=256)
u = env.unwrapped
an, rd, occ_an, occ_rd = [], [], [], []
n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 24
for i in range(n_scenes):
    split = ["iid", "occ_ood", "density_ood"][i % 3]
    cfg = sample_config(replace(SPLITS[split], dr=DR), 1000 + i)
    obs, _ = env.reset(seed=1000 + i, options=dict(cfg=cfg, which="plus"))
    seg = obs["sensor_data"]["base_camera"]["segmentation"][0, ..., 0].cpu().numpy()
    for k, a in enumerate(u.fruit_actor):
        sid = int(a.per_scene_id[0])
        cam = u.layout.cam
        f_px = 128 / np.tan(cam.fov / 2)
        d = np.linalg.norm(u.rest[k] - cam.eye)
        full = np.pi * (f_px * np.tan(np.arcsin(R_FRUIT / d))) ** 2
        vis_r = min(1.0, (seg == sid).sum() / full)
        an.append(u.vis[k]); rd.append(vis_r)
an, rd = np.array(an), np.array(rd)
np.savez(f"results/visibility_check_{DR}.npz", analytic=an, rendered=rd, scenes=n_scenes)
print(f"{len(an)} fruits from {n_scenes} scenes")
print(f"Pearson r = {np.corrcoef(an, rd)[0,1]:.3f} | mean |diff| = {np.abs(an-rd).mean():.3f} | median |diff| = {np.median(np.abs(an-rd)):.3f} | 90th pct |diff| = {np.quantile(np.abs(an-rd),0.9):.3f}")
for lo, hi in [(0, .3), (.3, .6), (.6, .9), (.9, 1.01)]:
    s = (an >= lo) & (an < hi)
    if s.any(): print(f"  analytic vis in [{lo:.1f},{min(hi,1):.1f}): n={s.sum():3d}  rendered mean {rd[s].mean():.2f}  (analytic mean {an[s].mean():.2f})")
