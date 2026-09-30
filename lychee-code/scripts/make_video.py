import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym, imageio, cv2
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()

seed = int(sys.argv[1]); out = sys.argv[2]
env = gym.make("LycheeProto-v0", obs_mode="rgb", control_mode="pd_ee_delta_pose", render_mode="rgb_array",
               sim_backend="cpu", render_backend="cpu")
u = env.unwrapped
obs, _ = env.reset(seed=seed)
ex = expert.Expert(env)
frames = []
def grab(obs):
    ext = env.render()
    ext = ext[0].cpu().numpy() if hasattr(ext, "cpu") else ext
    base = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
    hand = obs["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy()
    base = cv2.resize(base, (512, 512), interpolation=cv2.INTER_NEAREST)
    hand = cv2.resize(hand, (256, 256), interpolation=cv2.INTER_NEAREST)
    panel = np.concatenate([ext, base], axis=1)
    strip = np.zeros((256, 1024, 3), np.uint8)
    strip[:, :256] = hand
    txt = f'"{u.instruction}"   target #{u.target_idx}   phase: {{}}'
    return panel, strip, txt
for step in range(320):
    a = ex.act()
    obs, *_ = env.step(a)
    panel, strip, txt = grab(obs)
    strip = strip.copy()
    cv2.putText(strip, txt.format(ex.phase), (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(strip, f"step {step:3d}  n_fruit={u.n_fruit}  touched_nontarget={u.touched_nontarget}", (270, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(strip, "left: external view | right: policy camera (224px, upscaled) | hand cam", (270, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    frames.append(np.concatenate([panel, strip], axis=0))
    if ex.done: break
print("steps", step + 1, "success", bool(u.evaluate()["harvested"].item()), "detach_order", u.detach_order, "target", u.target_idx)
imageio.mimsave(out, frames, fps=10, macro_block_size=None)
print("wrote", out, len(frames), "frames")
