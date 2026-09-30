import sys, glob; sys.path.insert(0, ".")
import numpy as np, cv2, imageio
from lychee import record

path = sys.argv[1]
files = sorted(glob.glob(path)) if any(c in path for c in "*?") else [path]
palette = np.array([[0, 0, 0]] + [[(37 * k) % 255, (91 * k + 60) % 255, (151 * k + 120) % 255] for k in range(1, 17)], np.uint8)
n = 0
for f in files:
    for name, fr, meta in record.iter_episodes(f):
        T = len(fr["action"]); n += 1
        seg = fr["inst_seg"]; prim = meta["primary"]
        mask0 = (seg[0] == prim + 1).sum()
        pix = [(seg[0] == k + 1).sum() for k in range(meta["n_fruit"])]
        print(f"{name} T={T} '{meta['instruction']}' family={meta['family']} targets={meta['targets']} primary={prim} "
              f"success={meta['success']} first_frame_pixels(primary)={mask0} all={pix} vis(primary)={meta['vis'][prim]:.2f}")
        if n == 1:
            idx = [0, T // 3, 2 * T // 3, T - 1]
            row1 = np.concatenate([fr["base_rgb"][i] for i in idx], 1)
            row2 = np.concatenate([palette[fr["inst_seg"][i]] for i in idx], 1)
            row3 = np.concatenate([fr["hand_rgb"][i] for i in idx], 1)
            imageio.imwrite("results/frames/data_check.png", np.concatenate([row1, row2, row3], 0))
print("episodes:", n)
