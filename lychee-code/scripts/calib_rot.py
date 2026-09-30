import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
from scipy.spatial.transform import Rotation as Rot
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu", render_backend="cpu")
u = env.unwrapped; env.reset(seed=0)
ex = expert.Expert(env)
def tcp():
    return ex._tcp()
p0, R0 = tcp()
print("home rot err to R_TGT (rad):", np.round((expert.R_TGT * R0.inv()).as_rotvec(), 3), " tcp", np.round(p0, 3))
for axis in range(3):
    env.reset(seed=0)
    p0, R0 = tcp()
    a = np.zeros(7, dtype=np.float32); a[3 + axis] = 0.5; a[6] = 1
    for _ in range(4): env.step(a)
    p1, R1 = tcp()
    inc = (R1 * R0.inv()).as_rotvec()
    print(f"cmd +0.5 on rot axis {'xyz'[axis]} x4 steps -> world rotvec increment {np.round(inc,3)}  (expected +{0.5*0.1*4:.2f} on axis {'xyz'[axis]}), tcp shift {np.round(p1-p0,3)}")
