"""Damped-least-squares IK from a natural seed: Panda home with a horizontal gripper pointing +x."""
import warnings; warnings.filterwarnings("ignore")
import contextlib, io, sys, numpy as np, torch
import pytorch_kinematics as pk
from mani_skill import PACKAGE_ASSET_DIR

urdf = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    chain = pk.build_serial_chain_from_urdf(open(urdf, "rb").read(), end_link_name="panda_hand_tcp")
LOW = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

def rot_err(Rt, R):
    Re = Rt @ R.T
    v = torch.stack([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / 2
    s = v.norm(); c = (torch.trace(Re) - 1) / 2
    ang = torch.atan2(s, c)
    return v / s.clamp_min(1e-9) * ang if s > 1e-9 else torch.zeros(3)

def solve(seed, pos, Rt, iters=800, lam=0.05):
    q = torch.tensor(seed, dtype=torch.float32)
    for _ in range(iters):
        T = chain.forward_kinematics(q[None]).get_matrix()[0]
        err = torch.cat([torch.tensor(pos, dtype=torch.float32) - T[:3, 3], rot_err(Rt, T[:3, :3])])
        if err[:3].norm() < 1e-4 and err[3:].norm() < 1e-3:
            break
        J = chain.jacobian(q[None])[0]
        dq = J.T @ torch.linalg.solve(J @ J.T + lam**2 * torch.eye(6), err)
        q = torch.minimum(torch.maximum(q + 0.5 * dq, LOW), HIGH)
    T = chain.forward_kinematics(q[None]).get_matrix()[0]
    return q, (torch.tensor(pos, dtype=torch.float32) - T[:3, 3]).norm().item(), T

Rt = torch.tensor([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])   # tool z -> +x, fingers open along y
seeds = {"natural": [0.0, 0.3, 0.0, -1.9, 0.0, 2.2, 0.785], "elbow-up": [0.3, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785]}
for name, seed in seeds.items():
    for pos in [(0.265, 0.30, 0.62), (0.265, -0.30, 0.62), (0.30, 0.0, 0.70)]:
        q, e, T = solve(seed, pos, Rt)
        print(name, pos, "pos err %.4f" % e, "q =", np.round(q.numpy(), 3).tolist(),
              " z_tool", np.round(T[:3, 2].numpy(), 2))
