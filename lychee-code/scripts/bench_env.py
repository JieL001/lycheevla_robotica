import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, ".")
import numpy as np, torch, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()
torch.set_num_threads(1)

def bench(label, obs_mode, ctrl, n_ep=8, cams=None):
    kw = {} if cams is None else dict(sensor_configs=dict(width=cams, height=cams))
    env = gym.make("LycheeProto-v0", obs_mode=obs_mode, control_mode=ctrl, sim_backend="cpu", render_backend="cpu", **kw)
    u = env.unwrapped
    steps, t_reset, t0 = 0, 0.0, time.time()
    for seed in range(n_ep):
        t1 = time.time(); env.reset(seed=seed); t_reset += time.time() - t1
        if u.target_idx is None: continue
        ex = expert.Expert(env)
        for _ in range(200):
            env.step(ex.act()); steps += 1
            if ex.done: break
    dt = time.time() - t0
    print(f"{label:<46s} {steps/(dt-t_reset):6.1f} steps/s ({1000*(dt-t_reset)/steps:5.1f} ms/step) | reset {1000*t_reset/n_ep:5.0f} ms | {dt/n_ep:.2f} s/episode incl. planning")
    env.close()

bench("state obs (no camera readback), EE ctrl", "state", "pd_ee_delta_pose")
bench("rgb, 2 cams 224/128, EE ctrl", "rgb", "pd_ee_delta_pose")
bench("rgb+depth+seg, 2 cams 224/128, EE ctrl", "rgb+depth+segmentation", "pd_ee_delta_pose")
bench("rgb+depth+seg, base cam 256 (+hand 128)", "rgb+depth+segmentation", "pd_ee_delta_pose", cams=256)
