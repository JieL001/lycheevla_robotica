"""Closed-loop wrapper for the compact BC policy (runs on CPU inside the simulator process)."""
from __future__ import annotations

import numpy as np

from .evalkit import Policy


class BCPolicy(Policy):
    name = "bc"

    def __init__(self, ckpt: str, exec_steps: int = 4, mode: str = "normal"):
        import torch
        from .bc import BCNet, VOCAB
        torch.set_num_threads(1)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = BCNet(len(VOCAB), **ck.get("cfg", dict(film=ck["film"])))
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.pm = np.asarray(ck["prop_mean"], np.float32)
        self.ps = np.asarray(ck["prop_std"], np.float32)
        assert mode in ("normal", "blank", "gibberish", "swap")
        self.exec_steps, self.mode = exec_steps, mode
        tag = ckpt.replace("\\", "/").split("/")[-1]
        self.name = f"bc[{tag}|k={exec_steps}{'' if mode == 'normal' else '|' + mode}]"

    def reset(self, env, obs):
        """Language probes: blank command, random words, or the OTHER member's command of the pair (swap)."""
        from .bc import tokenize, VOCAB
        u = env.unwrapped
        if self.mode == "blank":
            text = ""
        elif self.mode == "gibberish":
            rng = np.random.RandomState(int(u.cfg.scene_seed) % (2 ** 31 - 1))
            words = [w for w in VOCAB if not w.startswith("<")]
            text = " ".join(rng.choice(words, size=6))
        elif self.mode == "swap" and u.cfg.minus is not None:
            text = (u.cfg.minus if u.which == "plus" else u.cfg.plus).text
        else:
            text = obs["instruction"]
        self.tok = tokenize(text)
        self.queue = []

    def act(self, obs):
        import torch
        from .preproc import preprocess
        from .bc import to_tensor_images
        if not self.queue:
            b, h = preprocess(obs["base_rgb"], obs["hand_rgb"])
            bt, ht = to_tensor_images(b[None], h[None], "cpu")
            p = torch.as_tensor(((obs["proprio"] - self.pm) / self.ps)[None], dtype=torch.float32)
            with torch.no_grad():
                chunk = self.net(bt, ht, p, torch.as_tensor(self.tok)[None])[0].numpy()
            self.queue = [np.clip(c, -1, 1).astype(np.float32) for c in chunk[: self.exec_steps]]
        return self.queue.pop(0)
