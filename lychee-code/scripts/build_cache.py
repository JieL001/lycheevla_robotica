"""HDF5 shards -> memory-mappable training cache (downscaled images).  Run in the `lychee` env (needs h5py + cv2).
   python scripts/build_cache.py --shards "D:/lychee_data/train_paired/shard_*.h5" --out D:/lychee_data/cache_paired
"""
import argparse, glob, json, os, sys, time
import numpy as np
from numpy.lib.format import open_memmap
import h5py
import cv2

sys.path.insert(0, ".")
from lychee import preproc, record
from lychee.bc import tokenize, MAXLEN

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
files = sorted(glob.glob(a.shards))
eps = []
for f in files:
    with h5py.File(f, "r") as fh:
        for name in fh:
            eps.append((f, name, int(fh[name].attrs["T"])))
N = sum(e[2] for e in eps)
print(f"{len(files)} shards, {len(eps)} episodes, {N} frames")
os.makedirs(a.out, exist_ok=True)
base = open_memmap(f"{a.out}/base.npy", "w+", np.uint8, (N, *preproc.BASE_OUT, 3))
hand = open_memmap(f"{a.out}/hand.npy", "w+", np.uint8, (N, *preproc.HAND_OUT, 3))
tmask = open_memmap(f"{a.out}/tmask.npy", "w+", np.float16, (N, 13, 20))
prop = np.zeros((N, 8), np.float32)
act = np.zeros((N, 7), np.float32)
meta, toks, t0, pos = [], [], time.time(), 0
by_file = {}
for f, name, T in eps:
    by_file.setdefault(f, []).append((name, T))
for f, items in by_file.items():
    with h5py.File(f, "r") as fh:
        for name, T in items:
            g = fh[name]
            b, h = g["base_rgb"][()], g["hand_rgb"][()]
            for t in range(T):
                base[pos + t], hand[pos + t] = preproc.preprocess(b[t], h[t])
            prop[pos:pos + T], act[pos:pos + T] = g["proprio"][()], g["action"][()]
            m = record.read_meta(g)
            seg = g["inst_seg"][()]
            tids = [int(t) + 1 for t in m["targets"]]
            for t in range(T):
                mk = np.isin(seg[t, preproc.BASE_ROWS[0]:preproc.BASE_ROWS[1]], tids).astype(np.float32)
                tmask[pos + t] = cv2.resize(mk, (20, 13), interpolation=cv2.INTER_AREA)
            meta.append(dict(name=name, start=pos, T=T, instruction=m["instruction"], family=m["family"], split=m["split"],
                             index=m["index"], which=m["which"], targets=m["targets"], primary=m["primary"],
                             success=bool(m["success"]), form=m["form"], edited_field=m.get("edited_field", "")))
            toks.append(tokenize(m["instruction"]))
            pos += T
    print(f"  {pos}/{N} frames  {time.time()-t0:.0f}s", flush=True)
base.flush()
hand.flush()
tmask.flush()
np.save(f"{a.out}/proprio.npy", prop)
np.save(f"{a.out}/action.npy", act)
np.save(f"{a.out}/tokens.npy", np.stack(toks))
json.dump(meta, open(f"{a.out}/episodes.json", "w"))
print("done", a.out)
