"""Evaluation kit: environment factory, policy interface, scripted baselines, episode / pair runners, metrics.

Policies see only what a learned VLA sees: ``base_rgb``, ``hand_rgb`` (uint8 HxWx3), ``proprio`` (8-D:
tcp position, tcp axis-angle, two gripper finger positions) and the instruction string.  The scripted
baselines below cheat on purpose (they read simulator state) -- they exist to bracket what is possible:

* ``expert``        goes for the commanded fruit (upper bound of the environment);
* ``blind_visible`` ignores language, always goes for the most visible fruit;
* ``blind_front``   ignores language, always goes for the fruit closest to the camera;
* ``attr_visible``  knows the commanded *maturity* but ignores every spatial word: most visible fruit of that
                    maturity (what a "grounds the colour word but not the relation" policy would do);
* ``rel_only``      knows the relation words (side / depth / ordinal) but ignores the maturity word;
* ``random``        a random fruit.

Pair metrics (PTA = paired target accuracy): a pair counts only if BOTH commands are answered correctly, so a
policy that ignores the instruction scores exactly 0 on pairs whose target sets are disjoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

MAX_STEPS = 300


def make_env(obs_mode: str = "state", img: int = 256, hand_img: int = 256, **kw):
    """The one place that fixes control mode, backends, IK and camera sizes, so data generation and evaluation
    always use identical dynamics."""
    import warnings
    warnings.filterwarnings("ignore")
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from . import env_proto, win_ik  # noqa: F401
    win_ik.patch()
    cams = dict(base_camera=dict(width=img, height=img), hand_camera=dict(width=hand_img, height=hand_img))
    return gym.make("LycheeProto-v0", obs_mode=obs_mode, control_mode="pd_ee_delta_pose", sim_backend="cpu",
                    render_backend="cpu", sensor_configs=cams, **kw)


def observation(env, obs) -> dict:
    """What a policy is allowed to see."""
    u = env.unwrapped
    sd = obs["sensor_data"]
    pose = u.agent.tcp.pose
    p = pose.p[0].cpu().numpy()
    q = pose.q[0].cpu().numpy()
    aa = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
    g = u.agent.robot.get_qpos()[0, -2:].cpu().numpy()
    return dict(base_rgb=sd["base_camera"]["rgb"][0].cpu().numpy(), hand_rgb=sd["hand_camera"]["rgb"][0].cpu().numpy(),
                proprio=np.concatenate([p, aa, g]).astype(np.float32), instruction=u.instruction)


# ------------------------------------------------------------------ policies
class Policy:
    name = "policy"

    def reset(self, env, obs: dict) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:            # normalised EE-delta action, shape (7,)
        raise NotImplementedError

    @property
    def done(self) -> bool:
        return False


class _ScriptedTarget(Policy):
    """Scripted expert redirected to a fruit chosen by ``choose``."""

    def choose(self, env) -> int:
        raise NotImplementedError

    def reset(self, env, obs):
        from .expert import Expert
        u = env.unwrapped                                    # the two members of a pair share scene_seed -> mix in `which`
        self.rng = np.random.RandomState((int(u.cfg.scene_seed) * 2 + (u.which == 'minus')) % (2 ** 31 - 1))
        self.chosen = self.choose(env.unwrapped)
        self.ex = Expert(env, target=self.chosen)

    def act(self, obs):
        return self.ex.act()

    @property
    def done(self):
        return self.ex.done


class ExpertPolicy(_ScriptedTarget):
    name = "expert"

    def choose(self, u):
        return u.target_idx


class BlindVisible(_ScriptedTarget):
    name = "blind_visible"

    def choose(self, u):
        return int(np.argmax(u.vis))


class BlindFront(_ScriptedTarget):
    name = "blind_front"

    def choose(self, u):
        return int(np.argmin([f.depth for f in u.fruit_views()]))


class AttrVisible(_ScriptedTarget):
    name = "attr_visible"

    def choose(self, u):
        cands = [i for i in range(u.n_fruit) if u.mat_of[i] == u.spec.mat]
        return max(cands, key=lambda i: u.vis[i]) if cands else int(np.argmax(u.vis))


class RelOnly(_ScriptedTarget):
    """Knows the relation words (side / depth / ordinal) but ignores the maturity word: applies the relation to ALL
    fruit (the mirror image of ``attr_visible``).  Commands without a relation (unique, any-of) get the most visible fruit."""
    name = "rel_only"

    def choose(self, u):
        s, fr = u.spec, u.fruit_views()
        if s.family == "mat_side":
            return min(fr, key=lambda f: f.u if s.side == "left" else -f.u).idx
        if s.family == "mat_depth":
            return min(fr, key=lambda f: f.depth if s.depth == "near" else -f.depth).idx
        if s.family == "mat_ordinal":
            order = sorted(fr, key=lambda f: f.u, reverse=(s.side == "right"))
            return order[min(s.k, len(order)) - 1].idx
        return int(np.argmax(u.vis))


class RandomFruit(_ScriptedTarget):
    name = "random"

    def choose(self, u):
        return int(self.rng.randint(u.n_fruit))


BASELINES = {c.name: c for c in (ExpertPolicy, BlindVisible, BlindFront, AttrVisible, RelOnly, RandomFruit)}


# ------------------------------------------------------------------ runners
def run_episode(env, policy: Policy, cfg, which: str = "plus", max_steps: int = MAX_STEPS) -> dict:
    u = env.unwrapped
    obs, _ = env.reset(seed=int(cfg.scene_seed) % (2 ** 31 - 1), options=dict(cfg=cfg, which=which))
    see = u.obs_mode != "state"
    o = observation(env, obs) if see else dict(instruction=u.instruction)
    policy.reset(env, o)
    t0, steps_since_detach = time.time(), 0
    for step in range(1, max_steps + 1):
        obs, *_ = env.step(policy.act(o))
        o = observation(env, obs) if see else dict(instruction=u.instruction)
        if u.detach_order:
            steps_since_detach += 1
        harvested = bool(u.evaluate().get("harvested", [False])[0])
        if policy.done or (harvested and steps_since_detach > 20) or steps_since_detach > 120:
            break
    info = {k: bool(v.item()) for k, v in u.evaluate().items()}
    ep = cfg.plus if which == "plus" else cfg.minus
    return dict(split=cfg.split, index=cfg.index, which=which, family=cfg.family, field=cfg.edited_field, form=ep.form,
                text=ep.text, n_fruit=u.n_fruit, steps=step, seconds=round(time.time() - t0, 2),
                first_detached=(u.detach_order[0] if u.detach_order else -1), first_grasped=(-1 if u.first_grasped is None else u.first_grasped),
                first_approached=(-1 if u.first_approached is None else u.first_approached),
                targets=list(ep.targets), primary=ep.primary, target_vis=float(min(u.vis[t] for t in ep.targets)),
                policy=policy.name, selected=int(getattr(policy, "chosen", -2)), **info)


def run_pair(env, policy: Policy, cfg) -> list:
    return [run_episode(env, policy, cfg, "plus"), run_episode(env, policy, cfg, "minus")]


# ------------------------------------------------------------------ metrics
def pair_table(records: list) -> dict:
    """{(split, index): {'plus': rec, 'minus': rec}} from flat episode records."""
    pairs = {}
    for r in records:
        pairs.setdefault((r["split"], r["index"]), {})[r["which"]] = r
    return {k: v for k, v in pairs.items() if "plus" in v and "minus" in v}


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion (k of n); non-degenerate at 0 and 1 (unlike the percentile bootstrap)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _prop(flags):
    """(share, Wilson lo, hi) of a list of booleans; the unit of the list is the scene (pair)."""
    flags = [bool(x) for x in flags]
    return wilson_ci(sum(flags), len(flags))


def _grasp_ok(r: dict) -> bool:
    """The first fruit the gripper closed on (stem still attached, fingers gripping) belongs to the target set."""
    g = r.get("first_grasped", -1)
    return g >= 0 and g in r.get("targets", [])


def _scene_mean(v: dict, fn) -> float:
    """Per-episode indicator averaged over the two members of a scene: the scene is the unit that is resampled."""
    return 0.5 * (float(fn(v["plus"])) + float(fn(v["minus"])))


def _chance_pta(v: dict) -> float:
    """PTA of a command-independent selector that picks uniformly among the fruit, independently for the two commands."""
    n = max(1, int(v["plus"].get("n_fruit", 1)))
    return len(v["plus"].get("targets", [])) * len(v["minus"].get("targets", [])) / (n * n)


def summarize(records: list, by: Optional[str] = None) -> dict:
    """Pair-level (PTA family, Wilson 95% intervals) and per-command metrics (percentile bootstrap over scenes, each scene
    contributing the mean of its two members).  ``by`` = a record key (e.g. 'family') to stratify by the plus member's value.
    Three-way outcome per command: out_correct (first detached fruit is a target), out_wrong (first detached fruit is not),
    out_none (nothing detached).  clean_success = success without touching a non-target fruit.  chance_PTA is a plain float:
    the mean PTA of a uniform random selector (independent draws per command) under the same scene/command sample."""
    pairs = pair_table(records)
    keys = sorted({str(v["plus"].get(by)) for v in pairs.values()}) if by else [None]
    out = {}
    for k in keys:
        sel = [v for v in pairs.values() if by is None or str(v["plus"].get(by)) == k]
        eps = [r for v in sel for r in (v["plus"], v["minus"])]
        sm = lambda fn: bootstrap_ci([_scene_mean(v, fn) for v in sel])
        both = lambda fn: _prop([fn(v["plus"]) and fn(v["minus"]) for v in sel])
        out[k if k is not None else "all"] = dict(
            n_pairs=len(sel),
            PTA_sel=both(lambda r: r["target_correct"]),
            PTA_succ=both(lambda r: r["success"]),
            PTA_appr=both(lambda r: r.get("approach_correct", False)),
            PTA_grasp=both(_grasp_ok),
            TSA_grasp=sm(_grasp_ok),
            grasp_rate=sm(lambda r: r.get("first_grasped", -1) >= 0),
            TSA=sm(lambda r: r["target_correct"]), TSA_appr=sm(lambda r: r.get("approach_correct", False)),
            success=sm(lambda r: r["success"]),
            clean_success=sm(lambda r: r["success"] and not r["touched_nontarget"]),
            wrong_target=sm(lambda r: r["wrong_target"]), touched=sm(lambda r: r["touched_nontarget"]),
            out_correct=sm(lambda r: r["target_correct"]),
            out_wrong=sm(lambda r: r["first_detached"] >= 0 and not r["target_correct"]),
            out_none=sm(lambda r: r["first_detached"] < 0),
            cond_success=bootstrap_ci([float(r["success"]) for r in eps if r["target_correct"]]),
            same_first_fruit=_prop([v["plus"]["first_detached"] == v["minus"]["first_detached"] and v["plus"]["first_detached"] >= 0 for v in sel]),
            chance_PTA=float(np.mean([_chance_pta(v) for v in sel])) if sel else float("nan"))
    return out


def macro_average(records: list, key: str = "PTA_sel", by: str = "family", n_boot: int = 2000, seed: int = 0):
    """Equal-weight mean of ``key`` over the groups of ``by`` with a stratified bootstrap over scenes (interval on the macro mean)."""
    pairs = pair_table(records)
    groups = {}
    for v in pairs.values():
        groups.setdefault(str(v["plus"].get(by)), []).append(v)
    flag = {"PTA_sel": lambda v: v["plus"]["target_correct"] and v["minus"]["target_correct"],
            "PTA_succ": lambda v: v["plus"]["success"] and v["minus"]["success"],
            "PTA_grasp": lambda v: _grasp_ok(v["plus"]) and _grasp_ok(v["minus"])}[key]
    vals = {g: np.array([float(flag(v)) for v in vs]) for g, vs in groups.items()}
    macro = float(np.mean([x.mean() for x in vals.values()])) if vals else float("nan")
    rng = np.random.RandomState(seed)
    boots = [np.mean([x[rng.randint(0, len(x), len(x))].mean() for x in vals.values()]) for _ in range(n_boot)]
    return macro, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), {g: len(x) for g, x in vals.items()}


def fmt(ci) -> str:
    m, lo, hi = ci
    return f"{100*m:5.1f} [{100*lo:5.1f},{100*hi:5.1f}]"
