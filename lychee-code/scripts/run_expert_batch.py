"""Batch-run the scripted expert; one JSON line per episode.  usage: seed0 n_eps nmin nmax out.jsonl"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, time; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()

seed0, n_eps, nmin, nmax, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu",
               render_backend="cpu", n_range=(nmin, nmax))
u = env.unwrapped
t0 = time.time()
with open(out, "w") as fh:
    for seed in range(seed0, seed0 + n_eps):
        env.reset(seed=seed)
        rec = dict(seed=seed, n=u.n_fruit, family=None if u.spec is None else u.spec.family)
        if u.target_idx is None:
            rec.update(status="no_valid_instruction"); fh.write(json.dumps(rec) + "\n"); continue
        ex = expert.Expert(env)
        steps = 0
        for steps in range(1, 321):
            env.step(ex.act())
            if ex.done: break
        fl = ex.flags
        success = bool(u.evaluate()["success"].item())          # target in the basket and nothing else detached (the metric of the paper)
        status = "ok" if success else ("no_grasp" if not fl["grasped"] else "no_detach" if not fl["detached"] else "lost")
        rec.update(status=status, steps=steps, touched=bool(u.touched_nontarget), gap_app=round(float(ex.gap_app), 3),
                   roll=float(ex.roll), vis_target=float(u.vis[u.target_idx]), min_vis=float(np.nanmin(u.vis)),
                   n_cf=len(u.cf), instr=u.instruction)
        fh.write(json.dumps(rec) + "\n"); fh.flush()
print(f"done {n_eps} episodes in {time.time()-t0:.0f}s", flush=True)
