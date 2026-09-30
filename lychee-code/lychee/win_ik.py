"""Make ManiSkill3's pd_ee_* controllers work with CPU simulation on Windows -- with a better IK step.

ManiSkill only uses pytorch_kinematics IK for CUDA tensors and falls back to sapien's pinocchio
bindings on CPU, which are missing from the Windows wheels.  ``patch()`` swaps in a Kinematics
subclass that (1) always uses pytorch_kinematics and (2) replaces ManiSkill's one-step
Levenberg-Marquardt update (damping 1e-4: it sacrifices orientation near wrist singularities) with
damped least squares on the 6x6 system, a null-space pull toward a nominal posture, and per-step
joint-motion clipping.  Call ``patch()`` before ``gym.make``.
"""
import torch
from lxml import etree as ET

from mani_skill.agents.controllers import pd_ee_pose
from mani_skill.agents.controllers.utils import kinematics as K
from mani_skill.utils.geometry import rotation_conversions as rc

Q_NOMINAL = torch.tensor([0.932, -0.844, 0.227, -2.536, -1.474, 2.075, -0.634])


def _rotvec_from_quat(q):                       # q: (B,4) wxyz, shortest-arc rotation vector
    q = torch.where(q[:, :1] < 0, -q, q)
    v = q[:, 1:]
    s = torch.linalg.norm(v, dim=1, keepdim=True)
    ang = 2 * torch.atan2(s, q[:, :1])
    return torch.where(s > 1e-9, v / s.clamp_min(1e-9) * ang, 2 * v)


class TorchKinematics(K.Kinematics):
    LAMBDA, K_NULL, MAX_DQ = 0.03, 0.4, 0.25

    def __init__(self, urdf_path, end_link_name, articulation, active_joint_indices):
        self.urdf_path = urdf_path
        self.end_link = articulation.links_map[end_link_name]
        self.articulation = articulation
        self.device = articulation.device
        self.active_joint_indices = active_joint_indices
        cur, chain = self.end_link, []
        while cur is not None:
            if cur.joint.active_index is not None:
                chain.append(cur.joint)
            cur = cur.joint.parent_link
        chain = chain[::-1]
        with open(urdf_path, "rb") as f:
            xml = ET.fromstring(f.read())
        self._kinematic_chain_joint_names = {n.get("name") for n in xml if n.tag == "joint"}
        self._kinematic_chain_link_names = {n.get("name") for n in xml if n.tag == "link"}
        self.active_ancestor_joints = [j for j in chain if j.name in self._kinematic_chain_joint_names]
        self._setup_gpu()

    def compute_ik(self, pose, q0, is_delta_pose=False, current_pose=None, solver_config=None):
        assert not is_delta_pose, "CPU path passes an absolute target pose"
        q = q0[:, self.active_ancestor_joint_idxs]                     # (B,7)
        dp = pose.p - current_pose.p
        dr = _rotvec_from_quat(rc.quaternion_multiply(pose.q, rc.quaternion_invert(current_pose.q)))
        err = torch.cat([dp, dr], dim=1).unsqueeze(-1)                 # (B,6,1)
        J = self.pk_chain.jacobian(q)                                  # (B,6,7)
        JT = J.transpose(1, 2)
        Jp = JT @ torch.linalg.inv(J @ JT + self.LAMBDA ** 2 * torch.eye(6, device=q.device))
        null = (torch.eye(7, device=q.device) - Jp @ J) @ (self.K_NULL * (Q_NOMINAL.to(q.device) - q)).unsqueeze(-1)
        dq = (Jp @ err + null).squeeze(-1).clamp(-self.MAX_DQ, self.MAX_DQ)
        return q + dq


def patch():
    pd_ee_pose.Kinematics = TorchKinematics
