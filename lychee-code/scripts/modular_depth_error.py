"""Quality of the depth that the modular baseline reads from the apparent radius of a detected fruit, against the 3 cm depth margin of the benchmark's rules.
   python scripts/modular_depth_error.py [--ckpts detector_s0 detector_s1 detector_s2] [--data D:/lychee_data/select_eval/iid_600] [--device cpu]
For every detection that is matched to a true fruit (lychee.detector.match_detections) it compares the depth inferred from the predicted radius with the depth inferred from the true
apparent radius (the inversion is exact for the true table: the oracle pipeline reproduces the labels).  Reports the median and the 90th percentile of the absolute radius error (px)
and depth error (cm), and the share of matched fruit whose depth error exceeds half of the 3 cm margin and the margin itself, pooled over the detector seeds.
Writes results/eval/modular_depth_error.md and ../tables/modular_depth_macros.tex (\\mdRadMed, \\mdRadNinety, \\mdDepthMed, \\mdDepthNinety, \\mdDepthOverHalf, \\mdDepthOverMargin)."""
import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from lychee.detector import FruitDetector, decode_detections, match_detections
from lychee.modular import depth_from_radius
from lychee.select import to_input

MARGIN_CM = 3.0


def errors(ck_path, data, dev, bs=64):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net = FruitDetector().to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    d = np.load(f"{data}/meta.npz")
    img = np.load(f"{data}/img.npy", mmap_mode="r")
    ok = d["ok"]
    rad_err, dep_err = [], []
    for s in range(0, len(ok), bs):
        idx = np.arange(s, min(len(ok), s + bs))
        with torch.no_grad():
            dets = decode_detections(net(to_input(np.stack([img[i] for i in idx]), dev)), ck["thr"])
        for det, i in zip(dets, idx):
            if not ok[i]:
                continue
            pairs, _, _ = match_detections(det, d["fxy"][i], d["fvis"][i], d["fvalid"][i])
            for j, f in pairs:
                rad_err.append(abs(float(det["rad"][j]) - float(d["frad"][i][f])))
                dep_err.append(100 * abs(depth_from_radius(float(det["rad"][j])) - depth_from_radius(float(d["frad"][i][f]))))
    return np.array(rad_err), np.array(dep_err)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="D:/lychee_data/select_ckpt")
    ap.add_argument("--ckpts", nargs="*", default=["detector_s0", "detector_s1", "detector_s2"])
    ap.add_argument("--data", default="D:/lychee_data/select_eval/iid_600")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    R, D = [], []
    for name in a.ckpts:
        p = f"{a.ckpt_dir}/{name}.pt"
        if not os.path.exists(p):
            continue
        r, dd = errors(p, a.data, a.device)
        R.append(r); D.append(dd)
        print(f"{name}: {len(r)} matched fruit, radius error median {np.median(r):.2f} px, depth error median {np.median(dd):.2f} cm, 90th percentile {np.percentile(dd, 90):.2f} cm", flush=True)
    R, D = np.concatenate(R), np.concatenate(D)
    st = dict(RadMed=np.median(R), RadNinety=np.percentile(R, 90), DepthMed=np.median(D), DepthNinety=np.percentile(D, 90),
              DepthOverHalf=100 * np.mean(D > MARGIN_CM / 2), DepthOverMargin=100 * np.mean(D > MARGIN_CM))
    md = [f"detector seeds pooled: {len(D)} matched fruit on {a.data}"] + [f"{k}: {v:.2f}" for k, v in st.items()]
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    open("results/eval/modular_depth_error.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    num = lambda k, v: f"{v:.2f}" if k.startswith("Rad") else f"{v:.1f}"             # radius errors are fractions of a pixel
    open("../tables/modular_depth_macros.tex", "w", newline="\n").write("".join(f"\\providecommand{{\\md{k}}}{{{num(k, v)}}}\n" for k, v in st.items()))
