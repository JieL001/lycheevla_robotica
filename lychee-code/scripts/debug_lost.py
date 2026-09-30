import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()
seed, nmin, nmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu", render_backend="cpu", n_range=(nmin, nmax))
u = env.unwrapped; env.reset(seed=seed); ex = expert.Expert(env); t = u.target_idx
prev = None
for step in range(320):
    env.step(ex.act())
    fp = u.fruit_actor[t].pose.p[0].numpy(); p, _ = ex._tcp()
    g = bool(u.agent.is_grasping(u.fruit_actor[t])[0])
    if ex.phase != prev or step % 10 == 0 or ex.phase in ("release",):
        print(f"{step:3d} {ex.phase:<9s} tcp={np.round(p,3)} fruit={np.round(fp,3)} grasp={g!s:<5} att={u.attached[t]!s:<5} basket_dist={np.linalg.norm(fp[:2]-u.BASKET_XY):.3f} latch={u.evaluate()['harvested'].item()}")
        prev = ex.phase
    if ex.done: break
print("final fruit", np.round(u.fruit_actor[t].pose.p[0].numpy(), 3), "latch", u.evaluate()["harvested"].item())
