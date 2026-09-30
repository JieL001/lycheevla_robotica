import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, gymnasium as gym
import mani_skill.envs
from lychee import win_ik, env_proto, expert; win_ik.patch()
seed = int(sys.argv[1])
env = gym.make("LycheeProto-v0", obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="cpu", render_backend="cpu")
u = env.unwrapped; env.reset(seed=seed)
ex = expert.Expert(env); t = u.target_idx
print("instr:", u.instruction, "target", t, "rest", np.round(u.rest[t], 3), "others:", [np.round(r, 3).tolist() for i, r in enumerate(u.rest) if i != t])
hand = u.agent.robot.links_map["panda_hand"]
for step in range(200):
    a = ex.act(); env.step(a)
    if ex.phase in ("approach", "grasp"):
        p, R = ex._tcp(); fp = u.fruit_actor[t].pose.p[0].numpy()
        f1 = float(np.linalg.norm(u.scene.get_pairwise_contact_forces(u.agent.finger1_link, u.fruit_actor[t]).numpy()))
        f2 = float(np.linalg.norm(u.scene.get_pairwise_contact_forces(u.agent.finger2_link, u.fruit_actor[t]).numpy()))
        print(f"{step:3d} {ex.phase:<8s} tcp-fruit={np.round(p-fp,3)} rot_err={np.round((expert.R_TGT*R.inv()).as_rotvec(),2)} finger_q={np.round(u.agent.robot.get_qpos()[0,-2:].numpy(),3)} contactF=({f1:.2f},{f2:.2f}) grasp={bool(u.agent.is_grasping(u.fruit_actor[t])[0])} hold={ex.hold}")
    if ex.phase == "pull": break
