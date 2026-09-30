"""Scripted EE-delta expert for LycheeProto (pre-grasp -> approach -> grasp -> pull -> transport -> release).

Actions are in ManiSkill's ``pd_ee_delta_pose`` space (normalised: 1 = 0.1 m / 0.1 rad per control step,
gripper +1 open / -1 closed), i.e. exactly the action space a VLA would be trained to output.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .approach import plan_approach, plan_pull

R_TGT = Rot.from_matrix([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])     # tool z -> +x, fingers open along y


class Expert:
    KP, KR = 1.6, 1.2
    PRE = np.array([-0.13, 0.0, 0.0])
    GRASP_OFF = np.array([-0.004, 0.0, 0.0])

    def __init__(self, env, plan=True, target=None):
        self.e = env.unwrapped
        t = self.e.target_idx if target is None else target
        self.t = t
        if plan:
            self.a, self.roll, self.R_tgt, self.gap_app = plan_approach(self.e.rest, t)
            self.pull_dir, self.gap_pull = plan_pull(self.e.rest, t, self.R_tgt)
        else:
            self.a, self.roll, self.R_tgt, self.gap_app = np.array([1.0, 0, 0]), 0.0, R_TGT, None
            self.pull_dir, self.gap_pull = self.e.PULL_DIR, None
        self.phase, self.n_in_phase, self.hold = "pregrasp", 0, 0
        self.history = []
        self.flags = dict(grasped=False, detached=False, dp_pre=None, dp_app=None)

    def _tcp(self):
        pose = self.e.agent.tcp.pose
        p, q = pose.p[0].cpu().numpy(), pose.q[0].cpu().numpy()
        return p, Rot.from_quat([q[1], q[2], q[3], q[0]])

    def _go(self, target, grip, p, R, kp=None, rt=None):
        rt = self.R_tgt if rt is None else rt
        a_pos = np.clip((self.KP if kp is None else kp) * (target - p) / 0.1, -1, 1)
        a_rot = -np.clip(self.KR * (rt * R.inv()).as_rotvec() / 0.1, -1, 1)   # ManiSkill: rot_action *= rot_lower (= -0.1)
        return np.concatenate([a_pos, a_rot, [grip]]).astype(np.float32), np.linalg.norm(target - p), np.linalg.norm((rt * R.inv()).as_rotvec())

    def _switch(self, name):
        self.phase, self.n_in_phase, self.hold = name, 0, 0

    def act(self) -> np.ndarray:
        e, t = self.e, self.t
        p, R = self._tcp()
        f = np.asarray(e.rest[t])
        self.n_in_phase += 1
        if self.phase == "pregrasp":
            a, dp, dr = self._go(f - 0.13 * self.a, 1.0, p, R)
            if (dp < 0.012 and dr < 0.15) or self.n_in_phase > 70:
                self.flags['dp_pre'] = round(float(dp), 3)
                self._switch("approach")
        elif self.phase == "approach":
            a, dp, dr = self._go(f - 0.004 * self.a, 1.0, p, R)
            if dp < 0.008 or self.n_in_phase > 40:
                self.flags['dp_app'] = round(float(dp), 3)
                self._switch("grasp")
        elif self.phase == "grasp":
            a, _, _ = self._go(f - 0.004 * self.a, -1.0, p, R)
            self.hold += bool(e.agent.is_grasping(e.fruit_actor[t])[0])
            if self.hold >= 4 or self.n_in_phase > 25:
                self.flags['grasped'] = self.hold >= 4
                self.anchor_p = p.copy()
                self._switch("pull")
        elif self.phase == "pull":
            a, _, _ = self._go(self.anchor_p + 0.07 * self.pull_dir, -1.0, p, R, kp=0.5)
            if not e.attached[t]:
                self.flags['detached'] = True
                self.hold += 1
            if self.hold >= 3 or self.n_in_phase > 40:
                self._switch("transport")
        elif self.phase == "transport":
            tgt = np.array([e.BASKET_XY[0], e.BASKET_XY[1], 0.20])
            a, dp, dr = self._go(tgt, -1.0, p, R, rt=R_TGT)
            if (dp < 0.03 and dr < 0.12) or self.n_in_phase > 90:
                self._switch("release")
        else:  # release
            a, _, _ = self._go(p, 1.0, p, R, rt=R_TGT)
            self.hold += 1
        self.history.append(self.phase)
        return a

    @property
    def done(self) -> bool:
        return self.phase == "release" and self.hold >= 14
