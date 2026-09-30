"""Generate expert demonstrations (HDF5 shards).  usage:
   python scripts/generate_data.py --split train --start 0 --n 200 --pair 1 --workers 4 --out D:/lychee_data/train
Paired mode (--pair 1) records BOTH members of a counterfactual pair on the same scene and keeps the pair only
if the expert succeeds on both; unpaired mode (--pair 0) records one instruction per scene; --pair 2 records two
INDEPENDENTLY drawn commands on the scene a pair would have used (control for the contrast).
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, ".")
_ENV, _WRITERS = None, {}


def _lower_priority():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _init():
    global _ENV
    _lower_priority()
    from lychee import evalkit
    _ENV = evalkit.make_env(obs_mode="rgb+segmentation")


def _record(env, cfg, which):
    from lychee import evalkit
    from lychee.expert import Expert
    u = env.unwrapped
    obs, _ = env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which=which))
    ex = Expert(env)
    ids = [int(a.per_scene_id[0]) for a in u.fruit_actor]
    fr = {k: [] for k in ("base_rgb", "hand_rgb", "inst_seg", "proprio", "action")}
    for step in range(evalkit.MAX_STEPS):
        o = evalkit.observation(env, obs)
        seg = obs["sensor_data"]["base_camera"]["segmentation"][0, ..., 0].cpu().numpy()
        inst = np.zeros(seg.shape, np.uint8)
        for k, sid in enumerate(ids):
            inst[seg == sid] = k + 1
        a = ex.act()
        fr["base_rgb"].append(o["base_rgb"]); fr["hand_rgb"].append(o["hand_rgb"]); fr["inst_seg"].append(inst)
        fr["proprio"].append(o["proprio"]); fr["action"].append(a)
        obs, *_ = env.step(a)
        if ex.done:
            break
    info = {k: bool(v.item()) for k, v in u.evaluate().items()}
    ep = cfg.plus if which == "plus" else cfg.minus
    lay = u.layout
    meta = dict(instruction=ep.text, family=ep.spec.family, mat=ep.spec.mat, side=ep.spec.side, depth=ep.spec.depth,
                k=ep.spec.k, targets=list(ep.targets), primary=ep.primary, form=ep.form, split=cfg.split, index=cfg.index,
                which=which, scene_seed=cfg.scene_seed, n_fruit=u.n_fruit, mats=[int(m) for m in lay.mats],
                fruit_pos=lay.pos.tolist(), vis=[float(v) for v in lay.vis], edited_field=cfg.edited_field or "",
                approach_roll=float(ex.roll), approach_gap=float(ex.gap_app), steps=len(fr["action"]), **info)
    return fr, meta, info


def _work(job):
    global _WRITERS
    split, index, pair, prefix = job
    from lychee import record
    from lychee.splits import sample_config
    t0 = time.time()
    try:
        cfg = sample_config(split, index, pair=bool(pair), indep=(pair == 2))
        which_list = ["plus", "minus"] if pair else ["plus"]
        recs = [_record(_ENV, cfg, w) for w in which_list]
    except Exception as e:
        return dict(index=index, status="error", err=repr(e))
    if not all(r[2]["success"] for r in recs):
        return dict(index=index, status="rejected", why=[r[2] for r in recs], seconds=time.time() - t0)
    pid = os.getpid()
    if pid not in _WRITERS:
        _WRITERS[pid] = record.ShardWriter(f"{prefix}_{pid}.h5")
    for (fr, meta, _), w in zip(recs, which_list):
        _WRITERS[pid].write(record.ep_name(split, index, w), fr, meta)
    return dict(index=index, status="ok", steps=[len(r[0]["action"]) for r in recs], seconds=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    jobs = [(a.split, i, int(a.pair), a.out) for i in range(a.start, a.start + a.n)]
    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
            res = pool.map(_work, jobs, chunksize=1)
    else:
        _init(); res = [_work(j) for j in jobs]
    ok = [r for r in res if r["status"] == "ok"]
    print(f"{a.split} pair={a.pair}: {len(ok)}/{len(res)} configs kept "
          f"({sum(r['status']=='rejected' for r in res)} rejected, {sum(r['status']=='error' for r in res)} errors), "
          f"{time.time()-t0:.0f}s, mean steps/episode {np.mean([s for r in ok for s in r['steps']]):.0f}")
    for r in res:
        if r["status"] == "error": print("  error:", r)
    json.dump(res, open(a.out + "_manifest.json", "w"), indent=0)
