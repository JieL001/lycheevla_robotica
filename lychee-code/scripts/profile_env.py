import warnings; warnings.filterwarnings("ignore")
import sys, time, cProfile, pstats; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto; win_ik.patch()
env = gym.make("LycheeProto-v0", obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose",
               render_mode="rgb_array", sim_backend="cpu", render_backend="cpu")
env.reset(seed=1)
a = np.zeros(7, dtype=np.float32)
for _ in range(5): env.step(a)
pr = cProfile.Profile(); pr.enable()
t0 = time.time()
for _ in range(30): env.step(a)
dt = time.time() - t0
pr.disable()
print(f"30 steps: {dt:.2f}s -> {30/dt:.1f} steps/s")
st = pstats.Stats(pr); st.sort_stats("cumulative").print_stats(18)
