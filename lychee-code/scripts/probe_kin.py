import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym, sapien
import mani_skill.envs
from lychee import win_ik, env_proto; win_ik.patch()
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="cpu", render_backend="cpu")
env.reset(seed=0); u = env.unwrapped
a = u.fruit_actor[0]
print("bodies:", type(a._bodies[0]), " has kinematic attr:", hasattr(a._bodies[0], "kinematic"))
b = a._bodies[0]
print("kinematic initially:", b.kinematic)
b.kinematic = True
print("kinematic after set:", b.kinematic)
z0 = float(a.pose.p[0, 2])
for _ in range(20): env.step(np.zeros(env.action_space.shape, dtype=np.float32))
print("kinematic fruit z drift over 20 steps:", float(a.pose.p[0, 2]) - z0)
b.kinematic = False
for _ in range(20): env.step(np.zeros(env.action_space.shape, dtype=np.float32))
print("dynamic fruit z change after release (should fall):", float(a.pose.p[0, 2]) - z0)
