import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, ".")
import numpy as np, torch, gymnasium as gym
import mani_skill.envs
from lychee import win_ik; win_ik.patch()

env = gym.make("PickCube-v1", obs_mode="state", control_mode="pd_ee_delta_pose",
               sim_backend="cpu", render_backend="none")
obs, _ = env.reset(seed=0)
u = env.unwrapped
tcp0 = u.agent.tcp.pose.p[0].clone()
print("action space", env.action_space)
print("tcp0", tcp0.numpy().round(4))
# +5 cm in z over 10 steps: action is normalised, |1| = pos_upper (0.1 m/step for pd_ee_delta_pose)
for _ in range(10):
    env.step(np.array([0, 0, 0.25, 0, 0, 0, -1.0], dtype=np.float32))
tcp1 = u.agent.tcp.pose.p[0]
print("tcp after 10 steps of dz=+0.25*0.1m:", tcp1.numpy().round(4), " dz =", float(tcp1[2]-tcp0[2]))
# lateral move and yaw
for _ in range(10):
    env.step(np.array([0.25, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
print("after +x:", u.agent.tcp.pose.p[0].numpy().round(4))
