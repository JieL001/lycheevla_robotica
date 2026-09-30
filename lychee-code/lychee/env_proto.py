"""ManiSkill3 selective-harvesting environment (pilot version of LycheeHarvest-Sim).

* The scene is built from a ``Layout`` (see ``layout.py``) and the command from an ``Episode`` (see
  ``splits.py``), so the SAME scene can be replayed with a different command:
  ``env.reset(seed=0, options=dict(cfg=pair_config, which="plus" | "minus"))``.
* "Virtual stem": every fruit is a dynamic body held at its rest position by a spring-damper plus gravity
  compensation; when it is grasped and stretched >= 2 cm along an admissible direction the stem "breaks"
  (spring removed, gravity restored) and the fruit is free.
* Visibility labels are analytic (ray casting, ``layout.fruit_visibility``); no extra renders at reset.
* Evaluation quantities (``evaluate``): ``target_correct`` (first fruit that came off the branch is one of
  the commanded fruits), ``wrong_target``, ``harvested``, ``success`` (correct fruit in the basket and nothing
  else detached), ``touched_nontarget`` (fingers/hand touched a non-target fruit).
Fruit are ~1.7x real size (radius 3.0 cm) so they span ~22 px at 256x256; leaves are visual-only.
On Windows CPU sim call ``lychee.win_ik.patch()`` before ``gym.make``.
"""
from __future__ import annotations

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env

from .layout import BRANCH, DEFAULT_CAM, R_FRUIT, sample_layout
from .splits import SplitConfig, sample_config

MAT_RGBA = {0: (0.35, 0.68, 0.15, 1.0), 1: (0.93, 0.66, 0.24, 1.0), 2: (0.78, 0.07, 0.10, 1.0)}
IMG = 256


@register_env("LycheeProto-v0", max_episode_steps=400)
class LycheeProtoEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: PandaWristCam

    R = R_FRUIT
    N_MAX, L_MAX = 16, 48
    BRANCH = BRANCH
    BASKET_XY = np.array([-0.15, -0.42])
    PULL_DIR = np.array([-0.35, 0.0, -0.94]) / np.linalg.norm([-0.35, 0.0, -0.94])
    PULL_DIST, PULL_COS = 0.02, np.cos(np.deg2rad(60.0))
    K_STEM, C_STEM = 300.0, 8.0                    # virtual stem: spring (N/m) and damper (N s/m)
    # IK solution (scripts/find_home.py): TCP at (-0.35, 0.30, 0.62) world, gripper horizontal, outside the camera FOV
    Q_HOME = np.array([0.932, -0.844, 0.227, -2.536, -1.474, 2.075, -0.634])

    def __init__(self, *args, robot_uids="panda_wristcam", n_range=(3, 10), leaf_range=(8, 30), min_vis=0.4, **kwargs):
        self.adhoc_split = SplitConfig("adhoc", n_range=n_range, leaf_range=leaf_range, target_vis=(min_vis, 1.0))
        self.fruit_pool = {0: [], 1: [], 2: []}
        self.leaves = []
        self.instruction, self.spec, self.cf = "", None, []
        self.targets, self.target_idx = set(), None
        self.layout, self.vis, self.rest, self.mat_of, self.n_fruit = None, None, [], [], 0
        self.fruit_actor, self.attached = [], []
        self.detach_order, self.first_grasped, self.basket_set = [], None, set()
        self.first_approached = None
        self.touched_nontarget = False
        self.cfg, self.which, self.episode = None, "plus", None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ sensors / robot
    @property
    def _default_sensor_configs(self):
        c = DEFAULT_CAM
        pose = sapien_utils.look_at(eye=c.eye.tolist(), target=c.target.tolist())
        return [CameraConfig("base_camera", pose=pose, width=IMG, height=IMG, fov=c.fov, near=0.01, far=4.0)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, -0.75, 0.75], target=[-0.05, 0.0, 0.32])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1.0, near=0.01, far=10)

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ------------------------------------------------------------------ scene
    def _load_scene(self, options):
        build_ground(self.scene)
        mat = lambda c: sapien.render.RenderMaterial(base_color=list(c), roughness=0.6)

        b = self.scene.create_actor_builder()
        b.add_cylinder_visual(radius=0.014, half_length=0.5, material=mat((0.30, 0.20, 0.10, 1)),
                              pose=sapien.Pose(q=[0.70710678, 0, 0, 0.70710678]))
        b.initial_pose = sapien.Pose(p=self.BRANCH.tolist())
        self.branch = b.build_kinematic("branch")

        m = self.scene.create_actor_builder()
        m.add_cylinder_visual(radius=0.07, half_length=0.002, material=mat((0.9, 0.9, 0.9, 1)),
                              pose=sapien.Pose(q=[0.70710678, 0, 0.70710678, 0]))
        m.initial_pose = sapien.Pose(p=[self.BASKET_XY[0], self.BASKET_XY[1], 0.002])
        self.basket = m.build_kinematic("basket_marker")

        for mt in range(3):
            self.fruit_pool[mt] = [
                actors.build_sphere(self.scene, radius=self.R, color=list(MAT_RGBA[mt]),
                                    name=f"fruit_m{mt}_{i}", body_type="dynamic",
                                    initial_pose=sapien.Pose(p=[0, 0, -3.0]))
                for i in range(self.N_MAX)
            ]
        rng = np.random.RandomState(0)
        self.leaves, self.leaf_base = [], []
        for i in range(self.L_MAX):
            g = rng.uniform(0.35, 0.65)
            lb = self.scene.create_actor_builder()
            lb.add_box_visual(half_size=[0.05, 0.022, 0.0008], material=mat((0.10, g, 0.08, 1)))
            lb.initial_pose = sapien.Pose(p=[0, 0, -4.0])
            self.leaves.append(lb.build_kinematic(f"leaf_{i}"))
            self.leaf_base.append((0.10, g, 0.08))

    # ------------------------------------------------------------------ episode
    def _initialize_episode(self, env_idx, options):
        options = options or {}
        cfg, which = options.get("cfg"), options.get("which", "plus")
        if cfg is None:                                        # ad-hoc episode: one unpaired instruction
            idx = int(self._episode_rng.randint(2 ** 30))
            cfg = sample_config(self.adhoc_split, idx, pair=False)
        ep = cfg.plus if which == "plus" else cfg.minus
        lay = sample_layout(cfg.scene_seed, **cfg.layout)
        self.layout, self.vis = lay, lay.vis
        n = lay.n

        self.fruit_actor, self.rest = [], []
        used = {0: 0, 1: 0, 2: 0}
        zero = torch.zeros(1, 3)
        for k in range(n):
            mt = int(lay.mats[k])
            a = self.fruit_pool[mt][used[mt]]
            used[mt] += 1
            b = a._bodies[0]
            b.kinematic = False
            b.disable_gravity = True                           # the virtual stem carries the weight
            a.set_pose(sapien.Pose(p=lay.pos[k].tolist()))
            a.set_linear_velocity(zero)
            a.set_angular_velocity(zero)
            self.fruit_actor.append(a)
            self.rest.append(lay.pos[k].copy())
        for mt in range(3):                                    # park unused actors below the ground
            for j in range(used[mt], self.N_MAX):
                self.fruit_pool[mt][j]._bodies[0].kinematic = True
                self.fruit_pool[mt][j].set_pose(sapien.Pose(p=[0.0, 0.0, -3.0 - 0.1 * j - mt]))
        self.mat_of = [int(x) for x in lay.mats]
        self.n_fruit = n
        self.attached = [True] * n
        self.detach_order, self.first_grasped, self.basket_set = [], None, set()
        self.first_approached = None
        self.touched_nontarget = False

        for i, lf in enumerate(self.leaves):
            if i < len(lay.leaves):
                l = lay.leaves[i]
                lf.set_pose(sapien.Pose(p=l.pos.tolist(), q=l.quat_wxyz.tolist()))
            else:
                lf.set_pose(sapien.Pose(p=[0, 0, -4.0 - 0.1 * i]))

        self._apply_visual_dr(lay)

        qpos = np.concatenate([self.Q_HOME + lay.q_noise, [0.04, 0.04]])
        self.agent.reset(torch.tensor(qpos, dtype=torch.float32)[None])
        self.agent.robot.set_pose(sapien.Pose(p=[-0.615, 0, 0]))

        self.cfg, self.which, self.episode = cfg, which, ep
        self.spec, self.instruction = ep.spec, ep.text
        self.targets, self.target_idx = set(ep.targets), ep.primary

    @staticmethod
    def _set_color(actor, rgb, alpha=1.0):
        m = actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent).render_shapes[0].material
        m.base_color = [float(np.clip(c, 0.0, 1.0)) for c in rgb] + [alpha]

    def _apply_visual_dr(self, lay):
        """Camera pose / field of view, ambient light and colour gains for this scene (see layout.DR)."""
        dr = lay.dr
        cam = self._sensors["base_camera"].camera
        cam.set_local_pose(sapien_utils.look_at(eye=lay.cam.eye.tolist(), target=lay.cam.target.tolist()).sp)
        cam.set_fovy(float(lay.cam.fov))
        self.scene.set_ambient_light((dr.ambient * dr.tint).tolist())
        for k in range(lay.n):
            mt = int(lay.mats[k])
            self._set_color(self.fruit_actor[k], np.array(MAT_RGBA[mt][:3]) * dr.fruit_gain)
        for i, lf in enumerate(self.leaves):
            self._set_color(lf, np.array(self.leaf_base[i]) * dr.leaf_gain)

    def fruit_views(self):
        return self.layout.fruit_views()

    # ------------------------------------------------------------------ virtual stem
    def _before_simulation_step(self):
        for i, a in enumerate(self.fruit_actor):
            if self.attached[i]:
                b = a._bodies[0]
                f = self.K_STEM * (self.rest[i] - np.asarray(b.pose.p)) - self.C_STEM * np.asarray(b.linear_velocity)
                b.add_force_torque(f.astype(np.float32), np.zeros(3, np.float32))

    def _after_control_step(self):
        tcp = self.agent.tcp.pose.p[0].cpu().numpy()
        if self.first_approached is None and self.rest:
            dist = np.linalg.norm(np.asarray(self.rest) - tcp, axis=1)
            if dist.min() < 0.05:                              # first fruit the fingertips come within 5 cm of
                self.first_approached = int(dist.argmin())
        for i, a in enumerate(self.fruit_actor):
            if not self.attached[i] or np.linalg.norm(tcp - self.rest[i]) > 0.09:
                continue
            grasped = bool(self.agent.is_grasping(a)[0])
            if grasped and self.first_grasped is None:
                self.first_grasped = i
            d = np.asarray(a._bodies[0].pose.p) - self.rest[i]
            n = np.linalg.norm(d)
            if grasped and n >= self.PULL_DIST and (d @ self.PULL_DIR) / n >= self.PULL_COS:
                self.attached[i] = False                       # stem breaks
                self.detach_order.append(i)
                a._bodies[0].disable_gravity = False
        if not self.touched_nontarget and self._touching_nontarget(tcp):
            self.touched_nontarget = True

    def _touching_nontarget(self, tcp) -> bool:
        if not self.targets:
            return False
        hand = self.agent.robot.links_map["panda_hand"]
        for i, a in enumerate(self.fruit_actor):
            if i in self.targets or np.linalg.norm(tcp - self.rest[i]) > 0.12 or not self.attached[i]:
                continue
            for link in (self.agent.finger1_link, self.agent.finger2_link, hand):
                if float(torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, a))) > 1e-3:
                    return True
        return False

    # ------------------------------------------------------------------ task
    def _in_basket(self, i: int) -> bool:
        p = self.fruit_actor[i].pose.p[0].cpu().numpy()
        return bool(np.linalg.norm(p[:2] - self.BASKET_XY) < 0.09 and p[2] < 0.15)

    def evaluate(self):
        """See the module docstring.  Everything is a bool tensor of shape (1,)."""
        b = lambda x: torch.tensor([bool(x)])
        if not self.targets:
            return dict(success=b(False))
        for i in self.detach_order:
            if not self.attached[i] and self._in_basket(i):
                self.basket_set.add(i)
        first = self.detach_order[0] if self.detach_order else -1
        wrong = any(i not in self.targets for i in self.detach_order)
        harvested = first in self.targets and first in self.basket_set
        return dict(success=b(harvested and not wrong), harvested=b(harvested), detached=b(bool(self.detach_order)),
                    target_correct=b(first in self.targets), wrong_target=b(wrong),
                    approach_correct=b(self.first_approached in self.targets),
                    touched_nontarget=b(self.touched_nontarget))

    def _get_obs_extra(self, info):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(1)

    def compute_normalized_dense_reward(self, obs, action, info):
        return torch.zeros(1)
