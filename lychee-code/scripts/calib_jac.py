import warnings; warnings.filterwarnings("ignore")
import contextlib, io, numpy as np, torch
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as Rot
from mani_skill import PACKAGE_ASSET_DIR
urdf = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    chain = pk.build_serial_chain_from_urdf(open(urdf, "rb").read(), end_link_name="panda_hand_tcp")
q0 = torch.tensor([[0.932, -0.844, 0.227, -2.536, -1.474, 2.075, -0.634]], dtype=torch.float32)
def fk(q):
    T = chain.forward_kinematics(q).get_matrix()[0].numpy()
    return T[:3, 3], Rot.from_matrix(T[:3, :3])
p0, R0 = fk(q0)
J = chain.jacobian(q0)[0]
print("J shape", tuple(J.shape))
for name, d in [("dx", [0.03, 0, 0, 0, 0, 0]), ("rot_x", [0, 0, 0, 0.05, 0, 0]), ("rot_y", [0, 0, 0, 0, 0.05, 0]), ("rot_z", [0, 0, 0, 0, 0, 0.05])]:
    d = torch.tensor(d, dtype=torch.float32)
    dq = torch.linalg.solve(J.T @ J + 1e-4 * torch.eye(7), J.T @ d)
    p1, R1 = fk((q0 + dq[None]))
    print(f"{name}: requested {np.round(d.numpy(),3)} -> achieved dp {np.round(p1-p0,3)}  world rotvec {np.round((R1*R0.inv()).as_rotvec(),3)}  body rotvec {np.round((R0.inv()*R1).as_rotvec(),3)}")
