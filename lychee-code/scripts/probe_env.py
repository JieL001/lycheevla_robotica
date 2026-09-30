import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, ".")
import numpy as np, torch, gymnasium as gym, imageio
import mani_skill.envs
from lychee import win_ik, env_proto; win_ik.patch()

env = gym.make("LycheeProto-v0", obs_mode="rgb+depth+segmentation", control_mode="pd_ee_delta_pose",
               render_mode="rgb_array", sim_backend="cpu", render_backend="cpu")
u = env.unwrapped
for seed in range(4):
    t0 = time.time()
    obs, _ = env.reset(seed=seed)
    print(f"[seed {seed}] reset {time.time()-t0:.2f}s  n_fruit={u.n_fruit}  mats={u.mat_of}")
    print("   instruction:", repr(u.instruction), "target", u.target_idx, "| n_valid_cf", len(u.cf))
    print("   vis:", np.round(u.vis, 2))
    rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
    imageio.imwrite(f"results/frames/probe_seed{seed}_base.png", rgb)
    imageio.imwrite(f"results/frames/probe_seed{seed}_hand.png", obs["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy())
    imageio.imwrite(f"results/frames/probe_seed{seed}_render.png", env.render()[0].cpu().numpy() if hasattr(env.render(), "cpu") else env.render())
print("tcp", u.agent.tcp.pose.p[0].numpy().round(3))
t0 = time.time(); n = 100
for _ in range(n):
    env.step(np.zeros(7, dtype=np.float32))
dt = time.time() - t0
print(f"{n} idle steps (rgb+depth+seg, 2 cams) in {dt:.2f}s -> {n/dt:.1f} steps/s")
