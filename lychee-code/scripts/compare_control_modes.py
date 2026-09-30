"""Does the expert work with target-relative EE deltas (actions accumulate exactly; better for open-loop chunks)?"""
import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()

mode = sys.argv[1]; n = int(sys.argv[2]); kp = float(sys.argv[3]); kr = float(sys.argv[4])
expert.Expert.KP, expert.Expert.KR = kp, kr
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode=mode, sim_backend="cpu", render_backend="cpu")
u = env.unwrapped
ok = tot = 0; steps = []; t0 = time.time()
for seed in range(n):
    env.reset(seed=seed)
    if u.target_idx is None: continue
    ex = expert.Expert(env); tot += 1
    for k in range(1, 321):
        env.step(ex.act())
        if ex.done: break
    good = bool(u.evaluate()["harvested"].item()) and not any(i != u.target_idx for i in u.detach_order)
    ok += good; steps.append(k)
    if not good: print(f"  fail seed {seed} phase {ex.phase} flags {ex.flags}")
print(f"{mode:<28s} KP={kp} KR={kr}: success {ok}/{tot}, mean steps {np.mean(steps):.0f}, {time.time()-t0:.0f}s")
