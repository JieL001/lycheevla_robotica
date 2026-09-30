import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu", render_backend="cpu")
u = env.unwrapped
env.reset(seed=seed)
ex = expert.Expert(env)
t = u.target_idx
print("instruction:", u.instruction, "| target", t, "at", np.round(u.rest[t], 3), "| n", u.n_fruit)
prev = None
for step in range(320):
    a = ex.act()
    env.step(a)
    p, R = ex._tcp()
    fp = u.fruit_actor[t].pose.p[0].numpy()
    g = bool(u.agent.is_grasping(u.fruit_actor[t])[0])
    qf = u.agent.robot.get_qpos()[0, -2:].numpy()
    if ex.phase != prev or step % 12 == 0:
        print(f"{step:3d} {ex.phase:<9s} tcp={np.round(p,3)} fruit={np.round(fp,3)} att={u.attached[t]!s:<5} grasp={g!s:<5} finger_q={np.round(qf,3)} a={np.round(a,2)}")
        prev = ex.phase
    if ex.done: break
print("success latch:", u.evaluate()["harvested"].item(), "| basket", u.BASKET_XY)
