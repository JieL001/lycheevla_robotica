"""A compact behaviour-cloning policy (trained from scratch) used as the small-model study.

Ways of feeding the command (all arms of the small-model comparison):
  * ``film=False, lang_direct=True``   late fusion: the command embedding is concatenated to the MLP head only
                                       (the classical design that is prone to ignore the command);
  * ``film=True``                      early fusion: every conv block is FiLM-modulated by the command;
  * ``ground=True``                    RS-GroundVLA-style path: a command-conditioned sigmoid heatmap over the
                                       main-camera feature map selects the top-k tokens, which enter the action
                                       head through a zero-initialised tanh-gated cross-attention block.  With
                                       ``lang_direct=False`` the command reaches the action ONLY through the
                                       grounded visual tokens (no language shortcut).  ``inject=False`` trains
                                       the heatmap but does not use it (supervision only); ``ground_sup=False``
                                       uses it without heatmap supervision (injection only).
Torch only; no simulator import.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lang import FRAMES, MAT_WORDS, SIDE_WORDS, DEPTH_WORDS, ORD_WORDS

CHUNK, ACT_DIM, PROPRIO_DIM, MAXLEN = 8, 7, 8, 12
FMAP_HW = (13, 20)              # main-camera feature-map size for the 100x160 input


def _words(text: str):
    return re.sub(r"[^a-z ]", " ", text.lower().replace("-", " ")).split()


def build_vocab() -> dict:
    words = set()
    for form in FRAMES.values():
        for frames in form.values():
            for f in frames:
                words.update(_words(f.replace("{m}", " ").replace("{s}", " ").replace("{d}", " ").replace("{o}", " ")))
    for form in MAT_WORDS.values():
        for ws in form.values():
            for w in ws:
                words.update(_words(w))
    for d in (SIDE_WORDS, DEPTH_WORDS):
        for form in d.values():
            for w in form.values():
                words.update(_words(w))
    for w in ORD_WORDS.values():
        words.update(_words(w))
    words.update(_words("without touching the fruits unripe green turning half ripe red"))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in sorted(words):
        vocab.setdefault(w, len(vocab))
    return vocab


VOCAB = build_vocab()


def tokenize(text: str, vocab: dict = VOCAB) -> np.ndarray:
    ids = [vocab.get(w, 1) for w in _words(text)][:MAXLEN]
    return np.array(ids + [0] * (MAXLEN - len(ids)), np.int64)


class SpatialSoftmax(nn.Module):
    def forward(self, x):                               # (B,C,H,W) -> (B,2C) expected keypoint coordinates
        B, C, H, W = x.shape
        p = F.softmax(x.flatten(2), dim=-1).view(B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
        return torch.cat([(p * xs).sum((2, 3)), (p * ys).sum((2, 3))], dim=1)


class Encoder(nn.Module):
    def __init__(self, chans, kernels, strides, cond_dim: int, film: bool):
        super().__init__()
        self.convs, self.norms, self.films = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c_in = 3
        for c, k, s in zip(chans, kernels, strides):
            self.convs.append(nn.Conv2d(c_in, c, k, s, k // 2))
            self.norms.append(nn.GroupNorm(8, c))
            self.films.append(nn.Linear(cond_dim, 2 * c) if film else None)
            c_in = c
        self.ss = SpatialSoftmax()
        self.c_out = c_in
        self.out_dim = 3 * c_in

    def forward(self, x, cond, return_map: bool = False):
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = norm(conv(x))
            if film is not None:
                g, b = film(cond).chunk(2, dim=1)
                x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
            x = F.relu(x)
        feat = torch.cat([self.ss(x), x.mean((2, 3))], dim=1)
        return (feat, x) if return_map else feat


class GroundHead(nn.Module):
    """Command-conditioned sigmoid heatmap over feature-map tokens; returns logits and the top-k grounded tokens."""

    def __init__(self, c_tok: int, e_dim: int = 128, k: int = 4, d: int = 64):
        super().__init__()
        self.k = k
        self.Wv, self.Wq, self.w = nn.Linear(c_tok + 2, d), nn.Linear(e_dim, d), nn.Linear(d, 1)

    def forward(self, fmap, e):
        B, C, H, W = fmap.shape
        ys = torch.linspace(-1, 1, H, device=fmap.device)
        xs = torch.linspace(-1, 1, W, device=fmap.device)
        coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1).view(1, H * W, 2).expand(B, -1, -1)
        z = torch.cat([fmap.flatten(2).transpose(1, 2), coords], dim=-1)             # (B,HW,C+2)
        s = self.w(torch.tanh(self.Wv(z) + self.Wq(e)[:, None])).squeeze(-1)         # (B,HW) logits
        top = s.topk(self.k, dim=1).indices
        G = torch.gather(z, 1, top[..., None].expand(-1, -1, z.shape[-1]))
        G = G * torch.sigmoid(torch.gather(s, 1, top))[..., None]                    # differentiable w.r.t. the heatmap
        return s.view(B, H, W), G


class GatedInject(nn.Module):
    """f' = f + tanh(gamma) * CrossAttn(f, G),  gamma_0 = 0 (identity at initialisation)."""

    def __init__(self, d_feat: int, c_tok: int, d: int = 128):
        super().__init__()
        self.q, self.k, self.v, self.o = nn.Linear(d_feat, d), nn.Linear(c_tok, d), nn.Linear(c_tok, d), nn.Linear(d, d_feat)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.d = d

    def forward(self, f, G):
        att = torch.softmax((self.q(f)[:, None] @ self.k(G).transpose(1, 2)) / math.sqrt(self.d), dim=-1)
        return f + torch.tanh(self.gamma) * self.o((att @ self.v(G))[:, 0])


class BCNet(nn.Module):
    def __init__(self, vocab_size: int, film: bool = True, hidden: int = 512, ground: bool = False,
                 inject: bool = True, lang_direct: bool = True, k: int = 4):
        super().__init__()
        self.film, self.ground, self.inject_on, self.lang_direct = film, ground, inject, lang_direct
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.base = Encoder([32, 64, 96, 128], [5, 3, 3, 3], [2, 2, 2, 1], 128, film)
        self.hand = Encoder([32, 64, 96], [5, 3, 3], [2, 2, 2], 128, film)
        self.prop = nn.Sequential(nn.Linear(PROPRIO_DIM, 64), nn.ReLU())
        d = self.base.out_dim + self.hand.out_dim + 64 + (128 if lang_direct else 0)
        if ground:
            self.ghead = GroundHead(self.base.c_out, 128, k)
            self.inj = GatedInject(d, self.base.c_out + 2)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, CHUNK * ACT_DIM))

    def encode_text(self, tok):
        lengths = (tok != 0).sum(1).clamp_min(1)
        out, _ = self.gru(self.emb(tok))
        return out[torch.arange(len(tok)), lengths - 1]

    def forward(self, base, hand, proprio, tok, return_ground: bool = False):
        e = self.encode_text(tok)
        if self.ground:
            fb, fmap = self.base(base, e, return_map=True)
        else:
            fb = self.base(base, e)
        parts = [fb, self.hand(hand, e), self.prop(proprio)] + ([e] if self.lang_direct else [])
        f = torch.cat(parts, dim=1)
        s = None
        if self.ground:
            s, G = self.ghead(fmap, e)
            if self.inject_on:
                f = self.inj(f, G)
        out = self.head(f).view(-1, CHUNK, ACT_DIM)
        return (out, s) if return_ground else out


def ground_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """Focal + Dice loss of the heatmap logits (B,H,W) against a soft target-coverage map (B,H,W) in [0,1]."""
    t = (target > 0.05).float()
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    focal = (alpha * t + (1 - alpha) * (1 - t)) * (1 - pt) ** gamma * bce
    inter = (p * t).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    return focal.mean() * 50 + dice.mean()


def to_tensor_images(base_u8: np.ndarray, hand_u8: np.ndarray, device):
    b = torch.as_tensor(base_u8, device=device).permute(0, 3, 1, 2).float() / 255.0
    h = torch.as_tensor(hand_u8, device=device).permute(0, 3, 1, 2).float() / 255.0
    return b, h


def random_shift_offsets(x: torch.Tensor, pad: int = 4):
    """Like random_shift, but also returns the shifts (B, 2) in pixels (x, y) by which the CONTENT of each image moved; a point at (px, py) in the input is at (px, py) + offset in the output."""
    B, C, H, W = x.shape
    sx = torch.randint(-pad, pad + 1, (B,), device=x.device).float()
    sy = torch.randint(-pad, pad + 1, (B,), device=x.device).float()
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = sx * 2 / W                                   # the sampling grid moves by +s, so the content moves by -s
    theta[:, 1, 2] = sy * 2 / H
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False), torch.stack([-sx, -sy], dim=1)


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = (torch.randint(-pad, pad + 1, (B,), device=x.device).float() * 2 / W)
    theta[:, 1, 2] = (torch.randint(-pad, pad + 1, (B,), device=x.device).float() * 2 / H)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)
