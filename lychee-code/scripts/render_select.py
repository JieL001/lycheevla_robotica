"""Render scenes and labels for the selection track (no manipulation is simulated, so this is ~20x cheaper than demonstrations).
   python scripts/render_select.py --split train --start 0 --n 4000 --mode pair --workers 5 --out D:/lychee_data/select/paired
   python scripts/render_select.py --split train --rho 0.9 --start 0 --n 8000 --mode single --out D:/lychee_data/select/rho90
   python scripts/render_select.py --split iid --start 0 --n 600 --mode pair --out D:/lychee_data/select_eval/iid_600
modes: pair = both commands of a counterfactual pair, single = one command, indep = two independently drawn commands on the pair's scene
(they name disjoint fruit in ~56% of the scenes), same = the second command paraphrases the first (no contrast at all).
--rho r (single mode): bias dial on the ``train`` distribution with a common seed range (100 M), so that all rho share their scenes.
Two stages: (1) labels (command sampling, fruit table; numpy only, many light workers), (2) images (simulator, few heavy workers).
Output directory: img.npy (N,160,256,3) uint8 memmap + fruit tables + tokens + target masks + meta.npz + info.json.
"""
import argparse, json, os, sys, time
import multiprocessing as mp
from dataclasses import replace

import numpy as np

sys.path.insert(0, ".")
FAMS = ["mat", "mat_any", "mat_side", "mat_depth", "mat_ordinal"]
_ENV = None


def _light():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _init():
    global _ENV
    _light()
    from lychee import evalkit
    _ENV = evalkit.make_env(obs_mode="rgb", img=256, hand_img=16)


def _label(job):
    """Stage 1 (no simulator): the config of one scene and every label derived from it."""
    split, index, mode = job
    from lychee.bc import tokenize
    from lychee.layout import R_FRUIT
    from lychee.select import fruit_table, N_MAX
    from lychee.splits import sample_config, layout_for
    try:
        cfg = sample_config(split, index, pair=(mode != "single"), indep=(mode == "indep"), same=(mode == "same"))
    except RuntimeError:
        return dict(index=index, ok=False)
    lay = layout_for(split, cfg.scene_seed)
    xy, rad, valid = fruit_table(lay, lay.pos, R_FRUIT)
    fmat = np.full(N_MAX, -1, np.int8)
    fvis = np.zeros(N_MAX, np.float32)
    fmat[:lay.n] = lay.mats
    fvis[:lay.n] = lay.vis
    eps = [cfg.plus] + ([cfg.minus] if cfg.minus is not None else [])
    tok = np.zeros((2, 12), np.int64)
    tmask = np.zeros((2, N_MAX), bool)
    primary = np.full(2, -1, np.int8)
    text = np.array(["", ""], dtype="<U96")
    for k, e in enumerate(eps):
        tok[k] = tokenize(e.text)
        tmask[k, list(e.targets)] = True
        primary[k] = e.primary
        text[k] = e.text
    return dict(index=index, ok=True, cfg=cfg, fxy=xy, frad=rad, fvalid=valid, fmat=fmat, fvis=fvis, tok=tok, tmask=tmask,
                primary=primary, ncmd=len(eps), family=FAMS.index(cfg.family), seed=int(cfg.scene_seed), text=text,
                field=cfg.edited_field or "", form=eps[0].form, n_fruit=int(lay.n))


def _render(cfg):
    """Stage 2: the third-person image of the scene of ``cfg`` (cropped rows 40:200)."""
    from lychee.select import CROP_ROWS
    obs, _ = _ENV.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which="plus"))
    return obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()[CROP_ROWS[0]:CROP_ROWS[1]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--rho", type=float, default=-1.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--mode", default="pair", choices=["pair", "single", "indep", "same"])
    ap.add_argument("--workers", type=int, default=4)             # simulator workers (stage 2)
    ap.add_argument("--label_workers", type=int, default=8)       # light workers (stage 1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from lychee.splits import SPLITS
    split = SPLITS[a.split]
    if a.rho >= 0:
        split = replace(SPLITS["train"], name=f"train_rho{int(round(a.rho * 100))}", bias_rho=a.rho, seed_base=100_000_000)
    os.makedirs(a.out, exist_ok=True)
    N = a.n
    arr = dict(fxy=np.zeros((N, 16, 2), np.float32), frad=np.zeros((N, 16), np.float32), fvalid=np.zeros((N, 16), bool),
               fmat=np.full((N, 16), -1, np.int8), fvis=np.zeros((N, 16), np.float32), tok=np.zeros((N, 2, 12), np.int64),
               tmask=np.zeros((N, 2, 16), bool), primary=np.full((N, 2), -1, np.int8), ncmd=np.zeros(N, np.int8),
               family=np.zeros(N, np.int8), seed=np.zeros(N, np.int64), text=np.zeros((N, 2), "<U96"), field=np.zeros(N, "<U8"),
               form=np.zeros(N, "<U1"), n_fruit=np.zeros(N, np.int8), ok=np.zeros(N, bool), index=np.arange(a.start, a.start + N))
    t0 = time.time()
    cfgs = [None] * N
    jobs = [(split, i, a.mode) for i in range(a.start, a.start + N)]
    with mp.get_context("spawn").Pool(a.label_workers, initializer=_light) as pool:
        for k, r in enumerate(pool.imap(_label, jobs, chunksize=4)):
            if r["ok"]:
                cfgs[k] = r.pop("cfg")
                for key in arr:
                    if key in r:
                        arr[key][k] = r[key]
                arr["ok"][k] = True
            if (k + 1) % 1000 == 0:
                print(f"  labels {k+1}/{N}  {time.time()-t0:.0f}s", flush=True)
    np.savez(f"{a.out}/meta.npz", **arr)
    print(f"labels done: {int(arr['ok'].sum())}/{N} scenes, {time.time()-t0:.0f}s", flush=True)
    img = np.lib.format.open_memmap(f"{a.out}/img.npy", "w+", np.uint8, (N, 160, 256, 3))
    todo = [(k, c) for k, c in enumerate(cfgs) if c is not None]
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for j, im in enumerate(pool.imap(_render, [c for _, c in todo], chunksize=2)):
            img[todo[j][0]] = im
            if (j + 1) % 500 == 0:
                print(f"  images {j+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    img.flush()
    json.dump(dict(split=split.name, start=a.start, n=N, mode=a.mode, rho=a.rho, ok=int(arr["ok"].sum()), seconds=time.time() - t0),
              open(f"{a.out}/info.json", "w"))
    print(f"done {a.out}: {int(arr['ok'].sum())}/{N} scenes in {time.time()-t0:.0f}s")
