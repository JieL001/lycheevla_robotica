"""Train the fruit detector (perception network without language) of the modular baseline on a rendered selection-track training set.  CUDA environment (relcomp, read-only use).
   python scripts/train_detector.py --data D:/lychee_data/select/rho00 --out D:/lychee_data/select_ckpt/detector_s0.pt --epochs 30 --seed 0
Same recipe as the selection networks (AdamW, one-cycle, batch 64, 30 epochs, 5 % of the scenes held out), no augmentation; the commands of the set are not used (the natural
unbiased set rho00 is the default).  After training the score threshold of the detector is chosen on the held-out scenes (maximum F1 of the detections, MATCH_PX) and stored
in the checkpoint."""
import argparse, os, sys, time

import numpy as np
import torch

sys.path.insert(0, ".")
from lychee.detector import FruitDetector, decode_detections, detector_loss, make_targets, match_detections
from lychee.select import to_input

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="D:/lychee_data/select/rho00")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.05)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"

d = np.load(f"{a.data}/meta.npz")
img = np.load(f"{a.data}/img.npy", mmap_mode="r")
scenes = np.flatnonzero(d["ok"])
rng = np.random.RandomState(a.seed)
perm = rng.permutation(scenes)
n_val = max(20, int(len(scenes) * a.val_frac))
val, tr = perm[:n_val], perm[n_val:]
fxy, frad, fmat, fvis, fvalid = (torch.as_tensor(d[k]) for k in ("fxy", "frad", "fmat", "fvis", "fvalid"))
print(f"scenes {len(scenes)} | train {len(tr)} | val {len(val)}", flush=True)

net = FruitDetector().to(dev)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps = max(1, len(tr) // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.1)


def batch(sc):
    order = np.argsort(sc)                                            # memmap gathers are faster in file order
    x = torch.as_tensor(np.stack([img[i] for i in sc[order]]))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return to_input(x.numpy(), dev)[inv]


def targets_of(sc):
    return make_targets(*(t[sc].to(dev) for t in (fxy, frad, fmat, fvis, fvalid)))


def detection_scores(thr):
    """Precision, recall, F1 of the detections on the held-out scenes and the maturity accuracy of the matched fruit."""
    net.eval(); tp = nd = ng = 0; mat_ok = 0
    with torch.no_grad():
        for s in range(0, len(val), 128):
            sc = val[s:s + 128]
            dets = decode_detections(net(batch(sc)), thr)
            for det, i in zip(dets, sc):
                pairs, n_det, n_gt = match_detections(det, d["fxy"][i], d["fvis"][i], d["fvalid"][i])
                tp += len(pairs); nd += n_det; ng += n_gt
                mat_ok += sum(int(det["mat"][j] == d["fmat"][i][f]) for j, f in pairs)
    net.train()
    p, r = tp / max(1, nd), tp / max(1, ng)
    return p, r, 2 * p * r / max(1e-9, p + r), mat_ok / max(1, tp)


t0 = time.time()
for ep in range(a.epochs):
    p = np.random.permutation(len(tr))
    run = 0.0
    for s in range(steps):
        sc = tr[p[s * a.bs:(s + 1) * a.bs]]
        loss = detector_loss(net(batch(sc)), targets_of(sc))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
    pr, rc, f1, macc = detection_scores(0.3)
    print(f"epoch {ep} loss {run/steps:.4f} val precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f} maturity acc {macc:.3f} ({time.time()-t0:.0f}s)", flush=True)
best = max(((detection_scores(t)[2], t) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)))
thr = best[1]
pr, rc, f1, macc = detection_scores(thr)
print(f"threshold {thr}: val precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f} maturity accuracy {macc:.3f}", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(dict(state=net.state_dict(), thr=thr, epochs=a.epochs, val=dict(precision=pr, recall=rc, f1=f1, maturity_acc=macc), args=vars(a)), a.out)
print("saved", a.out)
