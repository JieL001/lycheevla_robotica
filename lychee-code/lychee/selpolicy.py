"""Closed-loop wrapper of the selection network: the network picks the fruit from the third-person image and the command, the
scripted expert executes the pick.  The pixel-to-fruit rule uses the simulator's fruit centres and is identical for all arms."""
from __future__ import annotations

import numpy as np

from .evalkit import Policy


class SelectPolicy(Policy):
    name = "sel"

    def __init__(self, ckpt: str, mode: str = "normal"):
        import torch
        from .bc import VOCAB
        from .select import SelectNet
        torch.set_num_threads(1)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = SelectNet(len(VOCAB), cond=ck["cond"], late_dim=ck.get("args", {}).get("late_dim", 64))
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        assert mode in ("normal", "blank", "gibberish", "swap")
        self.mode = "blank" if ck.get("blank") else mode         # a network trained without commands always receives the empty command
        self.name = f"sel[{ckpt.replace(chr(92), '/').split('/')[-1]}{'' if mode == 'normal' else '|' + mode}]"
        self.ex, self.chosen, self.invalid = None, -1, True

    def reset(self, env, obs):
        import torch
        from .bc import tokenize, VOCAB
        from .expert import Expert
        from .select import CROP_ROWS, decode, fruit_table, to_input
        u = env.unwrapped
        if self.mode == "blank":
            text = ""
        elif self.mode == "gibberish":
            rng = np.random.RandomState(int(u.cfg.scene_seed) % (2 ** 31 - 1))
            text = " ".join(rng.choice([w for w in VOCAB if not w.startswith("<")], size=6))
        elif self.mode == "swap" and u.cfg.minus is not None:
            text = (u.cfg.minus if u.which == "plus" else u.cfg.plus).text
        else:
            text = obs["instruction"]
        x = to_input(obs["base_rgb"][CROP_ROWS[0]:CROP_ROWS[1]][None])
        xy, _, valid = fruit_table(u.layout, u.rest, float(u.R))
        with torch.no_grad():
            logits = self.net(x, torch.as_tensor(tokenize(text))[None])
            sel = int(decode(logits, torch.as_tensor(xy)[None], torch.as_tensor(valid)[None])[0])
        self.chosen, self.invalid = sel, sel < 0
        self.ex = None if self.invalid else Expert(env, target=sel)

    def act(self, obs):
        return np.zeros(7, np.float32) if self.invalid else self.ex.act()

    @property
    def done(self):
        return self.invalid or self.ex.done
