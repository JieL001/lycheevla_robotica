"""Items for the web-based human validation of the command labels (see scripts/human_check_template.html).
   python scripts/make_human_check_web.py --n 200 --out results/human_check_web
Writes items.json (what annotators see: image as JPEG data URI, command text, fruit positions in % of the image) and
key.json (targets, family, split -- NOT embedded in the page).  Items are stratified over splits and families and alternate
between the two members of counterfactual pairs; scene indices 500 and up are disjoint from every evaluation range.
"""
import argparse, base64, io, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

from lychee import evalkit
from lychee.select import fruit_table, IMG_HW
from lychee.splits import sample_config

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="results/human_check_web")
ap.add_argument("--start", type=int, default=500)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
env = evalkit.make_env(obs_mode="rgb", img=256, hand_img=16)
u = env.unwrapped
mix = ["iid"] * 5 + ["occ_ood"] * 2 + ["density_ood"] * 2 + ["saliency_rev"] * 1
items, key = [], []
idx = a.start
rng = np.random.RandomState(0)
fam_count = {}
while len(items) < a.n:
    split = mix[len(items) % len(mix)]
    try:
        cfg = sample_config(split, idx)
    except RuntimeError:
        idx += 1
        continue
    idx += 1
    which = "plus" if len(items) % 2 == 0 else "minus"
    ep = cfg.plus if which == "plus" else cfg.minus
    obs, _ = env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which=which))
    img = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()[40:200]
    xy, rad, valid = fruit_table(u.layout, u.rest, float(u.R))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=88)
    n = int(u.n_fruit)
    items.append(dict(id=len(items) + 1, text=ep.text, n=n,
                      fruit=[dict(k=j + 1, x=round(float(xy[j, 0]) / IMG_HW[1] * 100, 2), y=round(float(xy[j, 1]) / IMG_HW[0] * 100, 2), r=round(float(rad[j]), 1)) for j in range(n)],
                      img="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()))
    key.append(dict(item=len(items), split=split, index=cfg.index, which=which, text=ep.text, family=ep.spec.family,
                    targets=[int(t) + 1 for t in ep.targets], n_fruit=n, target_vis=float(min(u.layout.vis[t] for t in ep.targets))))
    fam_count[ep.spec.family] = fam_count.get(ep.spec.family, 0) + 1
json.dump(items, open(f"{a.out}/items.json", "w"))
json.dump(key, open(f"{a.out}/key.json", "w"), indent=1)
print(f"wrote {len(items)} items ({os.path.getsize(a.out + '/items.json')/1e6:.1f} MB); families {fam_count}")
