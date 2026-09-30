"""Qualitative figure: where do two selectors point for the two commands of a counterfactual pair?  Scenes are saliency-reversed test scenes on
which the strongly biased selector (R0, rho=0.97) fails and the paired selector (R1) succeeds.  Rows: scenes; columns: the scene with both
commands, then the pointing distribution (softmax over feature-map cells) of R1 and of R0 (rho=0.97) for each command.
   python scripts/make_pointing_figure.py [--n 3]   ->  ../figs/fig_pointing.pdf / .png"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from lychee.bc import VOCAB
from lychee.select import SelectNet, to_input, MAP_HW, CELL

DATA = "D:/lychee_data"
CK = f"{DATA}/select_ckpt"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "figure.dpi": 150})


def net_of(name):
    ck = torch.load(f"{CK}/{name}.pt", map_location="cpu", weights_only=False)
    net = SelectNet(len(VOCAB), cond=ck["cond"])
    net.load_state_dict(ck["state"])
    return net.eval()


def pair_ok(recs):
    pt = {}
    for r in recs:
        pt.setdefault(r["index"], {})[r["which"]] = r
    return {i: v["plus"]["target_correct"] and v["minus"]["target_correct"] for i, v in pt.items() if len(v) == 2}


def main(n):
    load = lambda tag: [json.loads(l) for l in open(f"results/eval/s_{tag}__sal.jsonl")]
    ok_a, ok_b = pair_ok(load("r1_film")), pair_ok(load("r0_rho97_film"))
    d = np.load(f"{DATA}/select_eval/sal_600/meta.npz")
    img = np.load(f"{DATA}/select_eval/sal_600/img.npy", mmap_mode="r")
    cand = [i for i in ok_a if ok_a[i] and not ok_b.get(i, True)]
    fam = d["family"]
    order, seen = [], set()
    for i in cand:                                          # one scene per command family first, relational families first
        f = int(fam[np.flatnonzero(d["index"] == i)[0]])
        if f in (3, 4) and f not in seen:
            order.append(i); seen.add(f)
    order += [i for i in cand if i not in order]
    picks = order[:n]
    nets = [("R1", net_of("r1_film")), (r"R0, $\rho=0.97$", net_of("r0_rho97_film"))]
    fig, axes = plt.subplots(len(picks), 5, figsize=(7.2, 1.55 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(picks):
        row = int(np.flatnonzero(d["index"] == idx)[0])
        x = to_input(np.asarray(img[row])[None])
        ax = axes[r, 0]
        ax.imshow(img[row]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("\n".join(f"{'+' if k == 0 else '-'} {d['text'][row, k]}" for k in range(2)), fontsize=5.5, loc="left")
        for c, (name, net) in enumerate(nets):
            for k in range(2):
                a = axes[r, 1 + 2 * c + k]
                tok = torch.as_tensor(d["tok"][row, k])[None]
                with torch.no_grad():
                    p = torch.softmax(net(x, tok), 1)[0].reshape(MAP_HW).numpy()
                a.imshow(img[row]); a.imshow(np.ma.masked_less(np.kron(p, np.ones((CELL, CELL))), 0.03), cmap="magma", alpha=0.8, vmin=0, vmax=1)
                for j in np.flatnonzero(d["tmask"][row, k]):
                    a.add_patch(Circle(d["fxy"][row, j], d["frad"][row, j] + 3, fill=False, ec="#00e5ff", lw=1.2))
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(f"{name}, command {'+' if k == 0 else '-'}", fontsize=6)
    fig.tight_layout(pad=0.3)
    os.makedirs("../figs", exist_ok=True)
    fig.savefig("../figs/fig_pointing.pdf"); fig.savefig("../figs/fig_pointing.png", dpi=200)
    print("wrote ../figs/fig_pointing.pdf with scenes", picks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    main(ap.parse_args().n)
