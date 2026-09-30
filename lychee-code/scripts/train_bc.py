"""Train the compact BC policy on a cache (run in an env with CUDA torch, e.g. relcomp -- read-only use).
   python scripts/train_bc.py --cache D:/lychee_data/cache_paired --out D:/lychee_data/ckpt/paired_film.pt --film 1
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")
from lychee.bc import BCNet, VOCAB, CHUNK, random_shift, to_tensor_images, ground_loss, tokenize


FMAP = (13, 20)


class FrameData(Dataset):
    def __init__(self, cache, episodes, blank=False):
        self.cache, self.episodes = cache, episodes
        self.blank_tok = tokenize("") if blank else None            # language-blind control: every command is replaced by the empty one
        self.starts = np.array([e["start"] for e in episodes])
        self.T = np.array([e["T"] for e in episodes])
        self.rows = np.array([e["_row"] for e in episodes])
        self.idx = np.concatenate([np.arange(e["start"], e["start"] + e["T"]) for e in episodes])
        self.ep_of = np.concatenate([np.full(e["T"], k) for k, e in enumerate(episodes)])
        self.m = None

    def _open(self):
        c = self.cache
        self.m = dict(base=np.load(f"{c}/base.npy", mmap_mode="r"), hand=np.load(f"{c}/hand.npy", mmap_mode="r"),
                      prop=np.load(f"{c}/proprio.npy", mmap_mode="r"), act=np.load(f"{c}/action.npy", mmap_mode="r"),
                      tok=np.load(f"{c}/tokens.npy"),
                      tm=np.load(f"{c}/tmask.npy", mmap_mode="r") if os.path.exists(f"{c}/tmask.npy") else None)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.m is None:
            self._open()
        g, k = int(self.idx[i]), int(self.ep_of[i])
        end = int(self.starts[k] + self.T[k])
        rows = np.minimum(np.arange(g, g + CHUNK), end - 1)
        return (np.array(self.m["base"][g]), np.array(self.m["hand"][g]), np.array(self.m["prop"][g]),
                self.blank_tok if self.blank_tok is not None else np.array(self.m["tok"][int(self.rows[k])]), np.array(self.m["act"][rows]),
                np.array(self.m["tm"][g], np.float32) if self.m["tm"] is not None else np.zeros(FMAP, np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--film", type=int, default=1)
    ap.add_argument("--ground", type=int, default=0)
    ap.add_argument("--inject", type=int, default=1)
    ap.add_argument("--ground_sup", type=int, default=1)
    ap.add_argument("--lang_direct", type=int, default=-1)      # -1: 1 without grounding, 0 with grounding
    ap.add_argument("--lambda_g", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--blank", type=int, default=0)                 # 1: train with the command blanked (language-blind reference)
    ap.add_argument("--max_eps", type=int, default=0)               # >0: use only the first N training episodes (data-scaling curve)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    meta = json.load(open(f"{a.cache}/episodes.json"))
    for r, e in enumerate(meta):
        e["_row"] = r
    rng = np.random.RandomState(a.seed)
    perm = rng.permutation(len(meta))
    n_val = max(4, len(meta) // 25)
    val_eps, tr_eps = [meta[i] for i in perm[:n_val]], [meta[i] for i in perm[n_val:]]
    if a.max_eps > 0:
        tr_eps = tr_eps[: a.max_eps]
    prop = np.load(f"{a.cache}/proprio.npy")
    pm, ps = prop.mean(0), prop.std(0) + 1e-3
    dev = "cuda"
    ld = (0 if a.ground else 1) if a.lang_direct < 0 else a.lang_direct
    cfg = dict(film=bool(a.film), ground=bool(a.ground), inject=bool(a.inject), lang_direct=bool(ld))
    print(f"train episodes used: {len(tr_eps)} | blank command: {bool(a.blank)}", flush=True)
    net = BCNet(len(VOCAB), **cfg).to(dev)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M | train eps {len(tr_eps)} val eps {len(val_eps)} | cfg={cfg}", flush=True)
    dl = DataLoader(FrameData(a.cache, tr_eps, bool(a.blank)), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0, pin_memory=True)
    vdl = DataLoader(FrameData(a.cache, val_eps, bool(a.blank)), batch_size=256, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)
    pm_t, ps_t = torch.tensor(pm, device=dev), torch.tensor(ps, device=dev)
    w = torch.tensor([1, 1, 1, 1, 1, 1, 2.0], device=dev)
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        net.train()
        run = 0.0
        for b, h, p, tok, act, tm in dl:
            b, h = to_tensor_images(b.numpy(), h.numpy(), dev)
            b, h = random_shift(b), random_shift(h)
            p = (p.to(dev) - pm_t) / ps_t
            tok = tok.to(dev)
            act = act.to(dev)
            out = net(b, h, p, tok, return_ground=True)
            loss = ((out[0] - act).abs() * w).mean()
            if a.ground and a.ground_sup:
                loss = loss + a.lambda_g * ground_loss(out[1], tm.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step % 200 == 0:
                print(f"  ep {ep} step {step}/{total} loss {run/200:.4f} {time.time()-t0:.0f}s", flush=True)
                run = 0.0
            if (time.time() - t0) / 60 > a.max_minutes:
                break
        net.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b, h, p, tok, act, tm in vdl:
                b, h = to_tensor_images(b.numpy(), h.numpy(), dev)
                l = ((net(b, h, (p.to(dev) - pm_t) / ps_t, tok.to(dev)) - act.to(dev)).abs() * w).mean()
                vs += l.item() * len(b)
                vn += len(b)
        print(f"epoch {ep} val L1 {vs/vn:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        torch.save(dict(state=net.state_dict(), film=bool(a.film), cfg=cfg, prop_mean=pm, prop_std=ps, epoch=ep, val=vs / vn), a.out)
        if (time.time() - t0) / 60 > a.max_minutes:
            break
    print("saved", a.out)
