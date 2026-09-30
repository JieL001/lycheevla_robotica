"""Print the physical constants the manuscript quotes, read from the running simulator (fruit mass, sim/control frequency, gripper)."""
import json, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lychee import evalkit
from lychee.splits import sample_config

env = evalkit.make_env(obs_mode="state")
u = env.unwrapped
cfg = sample_config("iid", 0)
env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which="plus"))
out = dict(
    fruit_mass_kg=[float(a._bodies[0].mass) for a in u.fruit_actor[:3]],
    fruit_radius_m=float(u.R),
    sim_freq_hz=int(u.sim_freq), control_freq_hz=int(u.control_freq),
    control_mode=str(env.unwrapped.control_mode),
    stem_K=u.K_STEM, stem_C=u.C_STEM, pull_dist_m=u.PULL_DIST, pull_cone_deg=60.0,
    n_fruit_actors=len(u.fruit_actor),
)
try:
    gp = u.agent.controller.controllers["gripper"].config
    out["gripper_limits"] = [float(gp.lower), float(gp.upper)]
    out["gripper_force_limit"] = str(getattr(gp, "force_limit", None))
except Exception as e:
    out["gripper_note"] = repr(e)
print(json.dumps(out, indent=1))
json.dump(out, open("results/sim_constants.json", "w"), indent=1)
