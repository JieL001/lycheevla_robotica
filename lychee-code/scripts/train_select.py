"""Train the selection network on a rendered dataset (see render_select.py).  Run in a CUDA environment (relcomp, read-only use).
   python scripts/train_select.py --data D:/lychee_data/select/train_paired --out D:/lychee_data/select_ckpt/r1_film_s0.pt --cond film --epochs 30
Options: --blank 1 (language-blind reference: the command is replaced by the empty one), --max_samples N (data scaling: whole scenes, so N samples
are N / 2 scenes of a paired set or N scenes of an unpaired set), --seed.
The N samples are (scene, command) pairs: a paired scene contributes both of its commands.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, ".")
from lychee.bc import VOCAB, tokenize, random_shift, random_shift_offsets
from lychee.select import SelectNet, SymbolicNet, target_distribution, select_loss, decode, to_input, fruit_features, symbolic_loss

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cond", default="film", choices=["film", "late", "token"])
ap.add_argument("--arch", default="pixel", choices=["pixel", "symbolic"])     # symbolic: privileged fruit table instead of pixels
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--blank", type=int, default=0)
ap.add_argument("--max_samples", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.05)
ap.add_argument("--scenes", type=int, default=0)            # use only the first K valid scenes (scene-binding dial)
ap.add_argument("--cmd0_only", type=int, default=0)         # 1: only the first command of each scene (one command per scene)
ap.add_argument("--total_steps", type=int, default=0)       # >0: fixed number of steps instead of epochs (batches sampled with replacement)
ap.add_argument("--shift", default="unsync", choices=["unsync", "sync", "none"])   # image shifts of up to 4 px: unsync = the labels are NOT shifted (all reported arms), sync = fruit coordinates shift with the image, none
ap.add_argument("--late_dim", type=int, default=64)             # width of the late-fusion scoring head (256: parameter count of the FiLM variant)
ap.add_argument("--far_keep", type=int, default=-1)         # >=0: keep only this many "farthest" commands (coverage control for the bias dial), all others stay
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"

d = np.load(f"{a.data}/meta.npz")
img = np.load(f"{a.data}/img.npy", mmap_mode="r")
ok, ncmd = d["ok"], d["ncmd"]
scenes = np.flatnonzero(ok)
if a.scenes > 0:
    scenes = scenes[: a.scenes]
rng = np.random.RandomState(a.seed)
perm = rng.permutation(scenes)
n_val = max(20, int(len(scenes) * a.val_frac))
val_scenes, tr_scenes = perm[:n_val], perm[n_val:]
samples = lambda sc: np.array([(i, k) for i in sc for k in range(1 if a.cmd0_only else int(ncmd[i]))])
tr, va = samples(tr_scenes), samples(val_scenes)
if a.max_samples > 0 and len(tr) > a.max_samples:
    # data scaling keeps WHOLE scenes: N samples = N / (commands per scene) scenes with all of their commands, so that a subset of the paired
    # set is still paired (a random subset of samples would rarely contain both commands of a scene)
    per = 1 if a.cmd0_only else max(1, int(round(len(tr) / max(1, len(tr_scenes)))))
    tr = samples(tr_scenes[: max(1, a.max_samples // per)])
if a.far_keep >= 0:
    # coverage control: the bias dial removes "farthest" commands (728 -> ~100 of 8,000); delete them at random from a NATURAL set instead of biasing the cue
    texts = d["text"]
    far = np.array([("farthest" in str(texts[i, k])) for i, k in tr])
    idx_far = np.flatnonzero(far)
    drop = set(rng.permutation(idx_far)[a.far_keep:].tolist())
    tr = tr[[j for j in range(len(tr)) if j not in drop]]
    print(f"far_keep {a.far_keep}: kept {min(a.far_keep, len(idx_far))} of {len(idx_far)} farthest commands", flush=True)
blank_tok = torch.as_tensor(tokenize(""))
print(f"scenes {len(scenes)} | train samples {len(tr)} | val samples {len(va)} | cond={a.cond} blank={bool(a.blank)}", flush=True)

sym = a.arch == "symbolic"
net = (SymbolicNet(len(VOCAB)) if sym else SelectNet(len(VOCAB), cond=a.cond, late_dim=a.late_dim)).to(dev)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps_per_epoch = max(1, len(tr) // a.bs)
if a.total_steps > 0:                                       # keep the number of updates comparable across arms with few scenes
    a.epochs, steps_per_epoch = max(1, a.total_steps // 100), 100
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
fxy, fvalid, tmask, tok = (torch.as_tensor(d[k]) for k in ("fxy", "fvalid", "tmask", "tok"))
feats_all = fruit_features(fxy, torch.as_tensor(d["frad"]), torch.as_tensor(d["fmat"]), torch.as_tensor(d["fvis"]), fvalid) if sym else None


def batch(idx, train: bool):
    sc, cm = idx[:, 0], idx[:, 1]
    if sym:
        t = torch.as_tensor(tok[sc, cm]) if not a.blank else blank_tok.expand(len(sc), -1).clone()
        return feats_all[sc].to(dev), t.to(dev), fxy[sc].to(dev), fvalid[sc].to(dev), tmask[sc, cm].to(dev)
    order = np.argsort(sc)                                            # memmap gathers are faster in file order
    sc_o = sc[order]
    x = torch.as_tensor(np.stack([img[i] for i in sc_o]))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    x = to_input(x.numpy(), dev)[inv]
    xy = fxy[sc].to(dev)
    if train and a.shift == "unsync":
        x = random_shift(x)
    elif train and a.shift == "sync":
        x, off = random_shift_offsets(x)
        xy = xy + off[:, None, :]                                          # the fruit centres move with the image
    t = torch.as_tensor(tok[sc, cm]) if not a.blank else blank_tok.expand(len(sc), -1).clone()
    return x, t.to(dev), xy, fvalid[sc].to(dev), tmask[sc, cm].to(dev)


def evaluate():
    net.eval(); hit, tot = 0, 0
    with torch.no_grad():
        for s in range(0, len(va), 128):
            x, t, xy, valid, tm = batch(va[s:s + 128], False)
            sel = net(x, t).argmax(1) if sym else decode(net(x, t), xy, valid)
            ok_ = (sel >= 0) & tm.gather(1, sel.clamp_min(0)[:, None]).squeeze(1)
            hit += int(ok_.sum()); tot += len(sel)
    net.train()
    return hit / tot


t0 = time.time()
for ep in range(a.epochs):
    p = np.random.permutation(len(tr)) if a.total_steps <= 0 else np.random.randint(0, len(tr), size=steps_per_epoch * a.bs)
    run = 0.0
    for s in range(steps_per_epoch):
        x, t, xy, valid, tm = batch(tr[p[s * a.bs:(s + 1) * a.bs]], True)
        loss = symbolic_loss(net(x, t), tm & valid) if sym else select_loss(net(x, t), target_distribution(xy, tm & valid))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
    acc = evaluate()
    print(f"epoch {ep} loss {run/steps_per_epoch:.4f} val selection accuracy {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(dict(state=net.state_dict(), cond=a.cond, arch=a.arch, blank=bool(a.blank), epochs=a.epochs, val_acc=acc, args=vars(a)), a.out)
print("saved", a.out)
