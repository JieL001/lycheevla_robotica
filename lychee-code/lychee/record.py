"""HDF5 episode recorder / reader for LycheeHarvest demonstrations.

One file (a "shard") holds many episodes; every episode is a group ``ep_<split>_<index>_<which>`` with

  base_rgb   (T,H,W,3) uint8   third-person camera (the policy's main view)
  hand_rgb   (T,H,W,3) uint8   wrist camera
  inst_seg   (T,H,W)   uint8   base-camera instance map: 0 = anything else, k+1 = fruit k
  proprio    (T,8)     float32 tcp position, tcp axis-angle, two finger positions
  action     (T,7)     float32 normalised EE-delta action (position 3, rotation 3, gripper)  [rotation sign follows ManiSkill]

and scene-level attributes (instruction text, command spec, target set, scene seed, fruit maturities /
positions / visibility, split, index, pair partner, success flags ...).  Images use LZF + shuffle.
"""
from __future__ import annotations

import json
from typing import Iterator

import h5py
import numpy as np

_IMG_KW = dict(compression="lzf", shuffle=True)


def ep_name(split: str, index: int, which: str) -> str:
    return f"ep_{split}_{index:07d}_{which}"


class ShardWriter:
    def __init__(self, path: str):
        self.fh = h5py.File(path, "a")

    def write(self, name: str, frames: dict, meta: dict) -> None:
        if name in self.fh:
            del self.fh[name]
        g = self.fh.create_group(name)
        T = len(frames["action"])
        for k in ("base_rgb", "hand_rgb"):
            a = np.asarray(frames[k], np.uint8)
            g.create_dataset(k, data=a, chunks=(1,) + a.shape[1:], **_IMG_KW)
        s = np.asarray(frames["inst_seg"], np.uint8)
        g.create_dataset("inst_seg", data=s, chunks=(1,) + s.shape[1:], compression="gzip", compression_opts=2, shuffle=True)
        g.create_dataset("proprio", data=np.asarray(frames["proprio"], np.float32))
        g.create_dataset("action", data=np.asarray(frames["action"], np.float32))
        for k, v in meta.items():
            v = "" if v is None else v                      # HDF5 attributes cannot hold None
            g.attrs[k] = json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
        g.attrs["T"] = T
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_meta(g) -> dict:
    out = {}
    for k, v in g.attrs.items():
        if isinstance(v, str) and v[:1] in "[{":
            try:
                v = json.loads(v)
            except ValueError:
                pass
        out[k] = v.item() if isinstance(v, np.generic) else v
    return out


def iter_episodes(path: str, load_images: bool = True) -> Iterator[tuple]:
    """Yield (name, frames dict, meta dict) for every episode in a shard."""
    with h5py.File(path, "r") as fh:
        for name in fh:
            g = fh[name]
            keys = ["proprio", "action"] + (["base_rgb", "hand_rgb", "inst_seg"] if load_images else [])
            yield name, {k: g[k][()] for k in keys}, read_meta(g)
