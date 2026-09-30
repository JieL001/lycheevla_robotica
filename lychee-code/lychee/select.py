"""Selection track: a language-conditioned pointing network for LycheeHarvest-Sim.

The learned policy answers only *which fruit*: it outputs a categorical distribution over the cells of a feature map of the
third-person image (CLIPort-style pointing); a fixed rule maps the arg-max cell to the nearest fruit centre and the scripted
expert executes the pick.  Manipulation is therefore identical for every arm, and PTA measures the selection alone.  The
end-to-end track (compact BC, VLA) is the harder companion; see README.

Torch only (no simulator import), so training runs in the CUDA environment and evaluation in either.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bc import Encoder, VOCAB, tokenize, MAXLEN            # noqa: F401  (re-exported)

CROP_ROWS = (40, 200)                 # rows of the 256x256 third-person image that contain the fruit
IMG_HW = (160, 256)                   # size of the crop fed to the network
CELL = 8                              # pixels per feature-map cell (three stride-2 convolutions)
MAP_HW = (IMG_HW[0] // CELL, IMG_HW[1] // CELL)      # (20, 32)
N_MAX = 16                            # fruit slots per scene
DECODE_RADIUS = 20.0                  # px: an arg-max cell further than this from every fruit centre is an invalid selection


def sincos_2d(h: int, w: int, d: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine position embedding, (h*w, d)."""
    assert d % 4 == 0
    q = d // 4
    freq = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32) / q))
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    ay, ax = ys.reshape(-1, 1) * freq, xs.reshape(-1, 1) * freq
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], dim=1)


class SelectNet(nn.Module):
    """cond = 'film'  : the command modulates every convolutional block (FiLM), linear pointing head;
       cond = 'late'  : language-agnostic image encoder, the command enters only at the pointing head (dot-product scoring);
       cond = 'token' : the command is an extra token inside the transformer, linear pointing head."""

    def __init__(self, vocab_size: int, cond: str = "film", d: int = 128, layers: int = 2, heads: int = 4, late_dim: int = 64):
        super().__init__()
        assert cond in ("film", "late", "token")
        self.cond, self.d = cond, d
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.enc = Encoder([32, 64, 96, d], [5, 3, 3, 3], [2, 2, 2, 1], 128, film=(cond == "film"))
        self.register_buffer("pos", sincos_2d(*MAP_HW, d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        if cond == "token":
            self.q_tok = nn.Linear(128, d)
        if cond == "late":
            self.Wv, self.Wq, self.w = nn.Linear(d, late_dim), nn.Linear(128, late_dim), nn.Linear(late_dim, 1)     # late_dim 256 makes the late variant as large as FiLM (0.61 M)
        else:
            self.lin = nn.Linear(d, 1)

    def encode_text(self, tok: torch.Tensor) -> torch.Tensor:
        lengths = (tok != 0).sum(1).clamp_min(1)
        out, _ = self.gru(self.emb(tok))
        return out[torch.arange(len(tok)), lengths - 1]

    def forward(self, img: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        """img (B,3,160,256) in [0,1]; tok (B,MAXLEN) -> logits (B, 20*32)."""
        e = self.encode_text(tok)
        _, fmap = self.enc(img, e, return_map=True)                   # (B,d,20,32)
        z = fmap.flatten(2).transpose(1, 2) + self.pos                # (B,640,d)
        if self.cond == "token":
            z = torch.cat([self.q_tok(e)[:, None], z], dim=1)
        z = self.tr(z)
        if self.cond == "token":
            z = z[:, 1:]
        if self.cond == "late":
            return self.w(torch.tanh(self.Wv(z) + self.Wq(e)[:, None])).squeeze(-1)
        return self.lin(z).squeeze(-1)


def cell_centres(device=None) -> torch.Tensor:
    """(640, 2) pixel coordinates (x, y) of the feature-map cell centres in the cropped image."""
    ys, xs = torch.meshgrid(torch.arange(MAP_HW[0], dtype=torch.float32), torch.arange(MAP_HW[1], dtype=torch.float32), indexing="ij")
    c = torch.stack([(xs + 0.5) * CELL, (ys + 0.5) * CELL], dim=-1).reshape(-1, 2)
    return c if device is None else c.to(device)


def target_distribution(fruit_xy: torch.Tensor, target_mask: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    """Soft target over cells: normalised mixture of Gaussians centred on the target fruit.  fruit_xy (B,N,2), target_mask (B,N) bool."""
    c = cell_centres(fruit_xy.device)                                              # (HW,2)
    d2 = ((fruit_xy[:, :, None, :] - c[None, None]) ** 2).sum(-1)                  # (B,N,HW)
    g = torch.exp(-d2 / (2 * sigma ** 2)) * target_mask[..., None].float()
    g = g.sum(1)
    return g / g.sum(1, keepdim=True).clamp_min(1e-8)


def select_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(1).mean()


def decode(logits: torch.Tensor, fruit_xy: torch.Tensor, fruit_valid: torch.Tensor, radius: float = DECODE_RADIUS) -> torch.Tensor:
    """Greedy selection: the fruit whose centre is nearest to the arg-max cell, or -1 if none lies within ``radius`` px."""
    c = cell_centres(logits.device)[logits.argmax(1)]                              # (B,2)
    d = ((fruit_xy - c[:, None]) ** 2).sum(-1).sqrt()                              # (B,N)
    d = d.masked_fill(~fruit_valid, float("inf"))
    dmin, idx = d.min(1)
    return torch.where(dmin <= radius, idx, torch.full_like(idx, -1))


def to_input(img_u8: np.ndarray, device=None) -> torch.Tensor:
    """(B,160,256,3) uint8 -> (B,3,160,256) float in [0,1]."""
    x = torch.as_tensor(img_u8).permute(0, 3, 1, 2).float() / 255.0
    return x if device is None else x.to(device)


def fruit_table(layout, rest, r_fruit: float):
    """Pixel-space fruit table of a scene in the cropped 160x256 frame: centres (N_MAX,2), radius (N_MAX,), valid (N_MAX,)."""
    cam = layout.cam
    f_px = (IMG_HW[1] / 2) / math.tan(cam.fov / 2)
    xy = np.zeros((N_MAX, 2), np.float32)
    rad = np.zeros(N_MAX, np.float32)
    valid = np.zeros(N_MAX, bool)
    for i, p in enumerate(np.asarray(rest)[:N_MAX]):
        u, depth, v = cam.project(p)
        xy[i] = (u * IMG_HW[1], v * IMG_HW[1] - CROP_ROWS[0])
        rad[i] = f_px * math.tan(math.asin(min(1.0, r_fruit / depth)))
        valid[i] = True
    return xy, rad, valid


# ----------------------------------------------------------------------------------------------------------------------
# Privileged symbolic reference: the same selection problem on the fruit table (no pixels).  It separates "the command is
# not understood" from "the fruit are not perceived": a symbolic selector that still follows the salience cue shows that the
# shortcut comes from the training data, not from perception.
# ----------------------------------------------------------------------------------------------------------------------
def fruit_features(fxy, frad, fmat, fvis, fvalid) -> torch.Tensor:
    """(B,N,8): x/256, y/160, radius/16 (a depth proxy), one-hot maturity (3), visibility, valid.  All arguments are tensors (B,N[,2])."""
    mat = torch.nn.functional.one_hot(fmat.clamp_min(0).long(), 3).float() * fvalid[..., None].float()
    return torch.cat([fxy[..., :1] / 256.0, fxy[..., 1:] / 160.0, (frad / 16.0)[..., None], mat, fvis[..., None], fvalid[..., None].float()], dim=-1)


class SymbolicNet(nn.Module):
    """Transformer over fruit tokens; the command is one extra token.  Output: one logit per fruit slot (invalid slots = -inf)."""

    def __init__(self, vocab_size: int, d: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.inp, self.q = nn.Linear(8, d), nn.Linear(128, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.lin = nn.Linear(d, 1)

    def encode_text(self, tok):
        lengths = (tok != 0).sum(1).clamp_min(1)
        out, _ = self.gru(self.emb(tok))
        return out[torch.arange(len(tok)), lengths - 1]

    def forward(self, feats: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        valid = feats[..., -1] > 0.5
        z = torch.cat([self.q(self.encode_text(tok))[:, None], self.inp(feats)], dim=1)
        mask = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1)              # True = ignore
        z = self.tr(z, src_key_padding_mask=mask)[:, 1:]
        return self.lin(z).squeeze(-1).masked_fill(~valid, float("-inf"))


def symbolic_loss(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    q = target_mask.float()
    q = q / q.sum(1, keepdim=True).clamp_min(1.0)
    logp = F.log_softmax(logits, dim=1)
    return -(q * logp.masked_fill(q == 0, 0.0)).sum(1).mean()
