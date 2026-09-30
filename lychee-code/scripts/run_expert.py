import warnings; warnings.filterwarnings("ignore")
import sys, time, collections; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()

n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PLAN = (sys.argv[2] != 'noplan') if len(sys.argv) > 2 else True
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu", render_backend="cpu")
u = env.unwrapped
rows, t0, steps_total = [], time.time(), 0
for seed in range(n_eps):
    env.reset(seed=seed)
    if u.target_idx is None:
        rows.append((seed, "no_valid_instruction", 0, "-", False)); continue
    ex = expert.Expert(env, plan=PLAN)
    for step in range(320):
        obs, r, te, tr, info = env.step(ex.act())
        steps_total += 1
        if ex.done: break
    ok = bool(u.evaluate()["harvested"].item()) and not any(i != u.target_idx for i in u.detach_order)
    last = ex.history[-1]
    fl = ex.flags
    reason = "ok" if ok else ("no_grasp" if not fl["grasped"] else "no_detach" if not fl["detached"] else "lost_in_transport") + ("+touch" if u.touched_nontarget else "")
    rows.append((seed, reason, step + 1, u.instruction, u.touched_nontarget))
    print(f"seed {seed:2d} n={u.n_fruit:2d} {reason:<22s} steps={step+1:3d} touched={u.touched_nontarget!s:<5} gap_app={ex.gap_app if ex.gap_app is None else round(ex.gap_app,3)} roll={ex.roll:.0f} | {u.instruction}", flush=True)
dt = time.time() - t0
ok = sum(1 for r in rows if r[1] == "ok"); valid = sum(1 for r in rows if r[1] != "no_valid_instruction")
print(f"\nsuccess {ok}/{valid} ({100*ok/max(1,valid):.0f}%), touched non-target {sum(r[4] for r in rows)}/{valid}, "
      f"{steps_total} steps in {dt:.0f}s ({steps_total/dt:.1f} steps/s)")
print(collections.Counter(r[1] for r in rows))
