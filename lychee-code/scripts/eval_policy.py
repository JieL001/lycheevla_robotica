"""Evaluate a policy on pair configs of a split.  usage:
   python scripts/eval_policy.py --policy expert --split iid --n 40 --workers 4 --out results/eval/expert_iid.jsonl
"""
import argparse, json, os, sys, time
import multiprocessing as mp

sys.path.insert(0, ".")
_ENV, _POLICY = None, None


def _lower_priority():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _init(policy_name, obs_mode):
    global _ENV, _POLICY
    _lower_priority()
    from lychee import evalkit
    _ENV = evalkit.make_env(obs_mode=obs_mode)
    if policy_name.startswith("bc|"):                          # bc|<ckpt path>|<exec_steps>[|blank]
        from lychee.bcpolicy import BCPolicy
        parts = policy_name.split("|")
        _POLICY = BCPolicy(parts[1], exec_steps=int(parts[2]) if len(parts) > 2 else 4, mode=parts[3] if len(parts) > 3 else "normal")
    elif policy_name.startswith("sel|"):                       # sel|<ckpt path>[|normal|blank|gibberish|swap]
        from lychee.selpolicy import SelectPolicy
        parts = policy_name.split("|")
        _POLICY = SelectPolicy(parts[1], mode=parts[2] if len(parts) > 2 else "normal")
    else:
        _POLICY = evalkit.BASELINES[policy_name]()


def _work(job):
    split, index = job
    from lychee import evalkit
    from lychee.splits import sample_config
    try:
        cfg = sample_config(split, index)
        return evalkit.run_pair(_ENV, _POLICY, cfg)
    except Exception as e:                      # keep the batch alive; report the failure
        return [dict(split=split, index=index, error=repr(e))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert"); ap.add_argument("--split", default="iid")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--obs", default="state")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from lychee import evalkit
    jobs = [(a.split, i) for i in range(a.start, a.start + a.n)]
    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.policy, a.obs)) as pool:
            res = pool.map(_work, jobs, chunksize=1)
    else:
        _init(a.policy, a.obs); res = [_work(j) for j in jobs]
    recs = [r for pair in res for r in pair if "error" not in r]
    errs = [r for pair in res for r in pair if "error" in r]
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            for r in recs: fh.write(json.dumps(r) + "\n")
    print(f"{a.policy} on {a.split}: {len(recs)//2} pairs, {len(errs)} errors, {time.time()-t0:.0f}s")
    for e in errs[:3]: print("  error:", e)
    for name, m in evalkit.summarize(recs).items():
        print(f"  n_pairs={m['n_pairs']}  PTA_sel {evalkit.fmt(m['PTA_sel'])}  PTA_succ {evalkit.fmt(m['PTA_succ'])}  PTA_appr {evalkit.fmt(m['PTA_appr'])}  "
              f"TSA {evalkit.fmt(m['TSA'])}  TSA_appr {evalkit.fmt(m['TSA_appr'])}  success {evalkit.fmt(m['success'])}  wrong {evalkit.fmt(m['wrong_target'])}  "
              f"touched {evalkit.fmt(m['touched'])}")
    for name, m in evalkit.summarize(recs, by="family").items():
        print(f"    {name:12s} n={m['n_pairs']:3d} PTA_sel {evalkit.fmt(m['PTA_sel'])} PTA_succ {evalkit.fmt(m['PTA_succ'])}")
