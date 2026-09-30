import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu", render_backend="cpu")
u = env.unwrapped; env.reset(seed=int(sys.argv[1]))
ex = expert.Expert(env); t = u.target_idx
for step in range(200):
    a = ex.act(); env.step(a)
    if ex.phase in ("grasp", "pull") and step < 130:
        p, _ = ex._tcp(); fp = u.fruit_actor[t].pose.p[0].numpy()
        print(f"{step:3d} {ex.phase:<6s} held={u.held} anchor={None if u.anchor[t] is None else np.round(u.anchor[t],3)} tcp={np.round(p,3)} fruit={np.round(fp,3)} "
              f"grasp={bool(u.agent.is_grasping(u.fruit_actor[t])[0])} att={u.attached[t]} finger={np.round(u.agent.robot.get_qpos()[0,-2:].numpy(),3)} a_pos={np.round(a[:3],2)}")
    if step > 125: break
