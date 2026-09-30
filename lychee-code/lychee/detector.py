"""Language-free perception network of the modular baseline (fruit detector + rule-based command parser, see lychee/modular.py).

It sees the same cropped third-person image (160x256 px) as the selection networks and has the same convolutional stem and two transformer layers, but no command.
Per cell of the 20x32 feature map it predicts a centre score (CenterNet-style heat map), the maturity (3 classes), the log radius in pixels (the apparent size, from which the
depth follows with the known camera and the known fruit radius), the sub-cell offset of the centre and the visible fraction.  Fruit whose visible fraction is below
MIN_VIS_GT are not detection targets (nothing of them is in the image).  Torch only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bc import Encoder
from .select import CELL, MAP_HW, N_MAX, cell_centres, sincos_2d

N_OUT = 1 + 3 + 1 + 2 + 1          # centre logit, maturity logits, log radius, offset (x, y), visibility logit
MIN_VIS_GT = 0.1                   # fruit less visible than this are not detection targets
SIGMA = 6.0                        # px, width of the Gaussian around a fruit centre in the heat-map target
MATCH_PX = 10.0                    # px, a detection matches a fruit if its centre is this close (evaluation of the detector itself)


class FruitDetector(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.enc = Encoder([32, 64, 96, d], [5, 3, 3, 3], [2, 2, 2, 1], 128, film=False)
        self.register_buffer("pos", sincos_2d(*MAP_HW, d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, N_OUT)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img (B,3,160,256) in [0,1] -> (B, 640, N_OUT)."""
        _, fmap = self.enc(img, torch.zeros(img.shape[0], 128, device=img.device), return_map=True)     # (B,d,20,32)
        return self.head(self.tr(fmap.flatten(2).transpose(1, 2) + self.pos))


def make_targets(fxy, frad, fmat, fvis, fvalid):
    """Training targets from the fruit table.  All arguments are tensors on one device: fxy (B,N,2) px, frad (B,N) px, fmat (B,N), fvis (B,N), fvalid (B,N) bool.
    Returns heat (B,640), pos (B,640) bool (the cell that contains a detectable fruit centre), and per-cell targets mat (B,640) long, lograd, offx, offy, vis (B,640)."""
    B, N = fmat.shape
    HW = MAP_HW[0] * MAP_HW[1]
    det = fvalid & (fvis >= MIN_VIS_GT)
    c = cell_centres(fxy.device)
    d2 = ((fxy[:, :, None, :] - c[None, None]) ** 2).sum(-1)                                   # (B,N,HW)
    heat = (torch.exp(-d2 / (2 * SIGMA ** 2)) * det[..., None]).max(1).values                  # (B,HW)
    cx = (fxy[..., 0] / CELL).floor().clamp(0, MAP_HW[1] - 1).long()
    cy = (fxy[..., 1] / CELL).floor().clamp(0, MAP_HW[0] - 1).long()
    cell = cy * MAP_HW[1] + cx                                                                  # (B,N)
    b_idx, n_idx = det.nonzero(as_tuple=True)
    c_idx = cell[b_idx, n_idx]
    heat[b_idx, c_idx] = 1.0
    pos = torch.zeros(B, HW, dtype=torch.bool, device=fxy.device)
    pos[b_idx, c_idx] = True
    mat = torch.zeros(B, HW, dtype=torch.long, device=fxy.device)
    lograd, offx, offy, vis = (torch.zeros(B, HW, device=fxy.device) for _ in range(4))
    mat[b_idx, c_idx] = fmat[b_idx, n_idx].long().clamp(0, 2)
    lograd[b_idx, c_idx] = torch.log(frad[b_idx, n_idx].clamp_min(1.0))
    offx[b_idx, c_idx] = fxy[b_idx, n_idx, 0] / CELL - cx[b_idx, n_idx].float() - 0.5
    offy[b_idx, c_idx] = fxy[b_idx, n_idx, 1] / CELL - cy[b_idx, n_idx].float() - 0.5
    vis[b_idx, c_idx] = fvis[b_idx, n_idx]
    return heat, pos, mat, lograd, offx, offy, vis


def detector_loss(out: torch.Tensor, targets) -> torch.Tensor:
    """CenterNet focal loss on the centre map plus masked regression / classification losses at the fruit cells."""
    heat, pos, mat, lograd, offx, offy, vis = targets
    p = out[..., 0].sigmoid().clamp(1e-4, 1 - 1e-4)
    n_pos = pos.sum().clamp_min(1)
    loss_pos = -(((1 - p) ** 2) * torch.log(p))[pos].sum()
    loss_neg = -(((1 - heat) ** 4) * (p ** 2) * torch.log(1 - p))[~pos].sum()
    loss = (loss_pos + loss_neg) / n_pos
    if pos.any():
        o = out[pos]
        loss = loss + F.cross_entropy(o[:, 1:4], mat[pos]) + F.l1_loss(o[:, 4], lograd[pos]) \
            + F.l1_loss(o[:, 5], offx[pos]) + F.l1_loss(o[:, 6], offy[pos]) + F.binary_cross_entropy_with_logits(o[:, 7], vis[pos])
    return loss


def decode_detections(out: torch.Tensor, thr: float = 0.3, k_max: int = N_MAX):
    """Peaks of the centre map (3x3 non-maximum suppression, score > thr, at most k_max) -> one dict of numpy arrays per scene:
    x, y (px in the cropped frame), rad (px), mat (0..2), vis (0..1), score."""
    B = out.shape[0]
    H, W = MAP_HW
    heat = out[..., 0].sigmoid().view(B, H, W)
    keep = (heat >= F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]) & (heat > thr)
    res = []
    out_cpu = out.detach().float().cpu()
    heat_cpu, keep_cpu = heat.detach().cpu(), keep.cpu()
    for b in range(B):
        cells = keep_cpu[b].flatten().nonzero().flatten()
        if len(cells) > k_max:
            cells = cells[heat_cpu[b].flatten()[cells].topk(k_max).indices]
        o = out_cpu[b, cells]
        cy, cx = cells // W, cells % W
        res.append(dict(x=((cx.float() + 0.5 + o[:, 5]) * CELL).numpy(), y=((cy.float() + 0.5 + o[:, 6]) * CELL).numpy(),
                        rad=o[:, 4].exp().numpy(), mat=o[:, 1:4].argmax(1).numpy(), vis=o[:, 7].sigmoid().numpy(),
                        score=heat_cpu[b].flatten()[cells].numpy()))
    return res


def detections_from_table(fxy, frad, fmat, fvis, fvalid):
    """The 'perfect perception' detections of one scene (numpy (N,2), (N,), (N,), (N,), (N,) bool): every valid fruit, exact attributes (used for the oracle run and tests)."""
    v = np.asarray(fvalid, bool)
    return dict(x=np.asarray(fxy)[v, 0].astype(float), y=np.asarray(fxy)[v, 1].astype(float), rad=np.asarray(frad)[v].astype(float),
                mat=np.asarray(fmat)[v].astype(int), vis=np.asarray(fvis)[v].astype(float), score=np.asarray(fvis)[v].astype(float))


def match_detections(det, fxy, fvis, fvalid, max_px: float = MATCH_PX):
    """Greedy nearest matching of detections to detectable fruit (fvis >= MIN_VIS_GT); returns (matched pairs [(det idx, fruit idx)], n_det, n_gt)."""
    gt = [i for i in range(len(fvalid)) if fvalid[i] and fvis[i] >= MIN_VIS_GT]
    used, pairs = set(), []
    order = np.argsort(-det["score"]) if len(det["score"]) else []
    for j in order:
        best, bd = None, max_px
        for i in gt:
            if i in used:
                continue
            dd = float(np.hypot(det["x"][j] - fxy[i][0], det["y"][j] - fxy[i][1]))
            if dd <= bd:
                best, bd = i, dd
        if best is not None:
            used.add(best)
            pairs.append((int(j), best))
    return pairs, len(det["score"]), len(gt)
