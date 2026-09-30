"""Offline selection-only evaluation of a selection network on a rendered evaluation set (render_select.py --mode pair).
   python scripts/eval_select.py --ckpt D:/lychee_data/select_ckpt/r1_film_s0.pt --data D:/lychee_data/select_eval/iid_600 --out results/select/r1_film_s0__iid.jsonl [--mode swap]
The scripted expert executes the pick with 99% success, so the record treats the selected fruit as the first detached fruit
(mode field 'selection'); closed-loop agreement is checked separately with eval_policy.py --policy "sel|<ckpt>".
Probes: --mode normal | blank | gibberish | swap (the command of the other member of the pair).  A checkpoint trained without commands (--blank 1 in train_select.py) always
receives the empty command, in every mode (see command_tokens).
Records have the fields of evalkit.run_episode, so evalkit.summarize / compare_policies work unchanged.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, ".")
from lychee.bc import VOCAB, tokenize
from lychee.select import DECODE_RADIUS, SelectNet, SymbolicNet, decode, to_input, fruit_features

FAMS = ["mat", "mat_any", "mat_side", "mat_depth", "mat_ordinal"]


def load_net(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = (SymbolicNet(len(VOCAB)) if ck.get("arch") == "symbolic" else SelectNet(len(VOCAB), cond=ck["cond"], late_dim=ck.get("args", {}).get("late_dim", 64))).to(dev)
    net.load_state_dict(ck["state"])
    return net.eval(), ck


def command_tokens(mode, ck, tok, idx, k, rng, words, blank, raw=False):
    """Command tokens that the network receives for scenes idx and command slot k.
    A checkpoint trained without commands (ck['blank']) is the language-blind reference: its command input is the EMPTY command in every mode.  A swapped or gibberish
    command would be an input that the network never saw, and 'language sensitivity' (TSA normal minus TSA swapped) would then measure its reaction to that input instead of
    being 0 by construction (the audit of 9/30 found 2.3 points for R1-blank; results/eval/blank_probe_audit.md).  raw=True bypasses the rule (diagnostic of that reaction)."""
    import torch
    if mode == "blank" or (ck.get("blank") and not raw):
        return blank.expand(len(idx), -1).clone()
    if mode == "gibberish":
        return torch.as_tensor(np.stack([tokenize(" ".join(rng.choice(words, size=6))) for _ in idx]))
    if mode == "swap":
        return torch.as_tensor(tok[idx, 1 - k])
    return torch.as_tensor(tok[idx, k])


def make_records(d, sel, ok, ncmd, policy, split_name):
    """Episode records (the fields of evalkit.run_episode) from the selected fruit index sel[i, k] (-1 = invalid) of every scene i and command k."""
    tmask, text = d["tmask"], d["text"]
    recs = []
    for i in np.flatnonzero(ok):
        for k, which in enumerate(("plus", "minus")[: int(ncmd[i])]):
            targets = [int(j) for j in np.flatnonzero(tmask[i, k])]
            s_ = int(sel[i, k])
            correct = s_ >= 0 and s_ in targets
            recs.append(dict(split=split_name, index=int(d["index"][i]), which=which, family=FAMS[int(d["family"][i])],
                             field=str(d["field"][i]), form=str(d["form"][i]), text=str(text[i, k]), n_fruit=int(d["n_fruit"][i]),
                             steps=0, seconds=0.0, first_detached=s_, first_grasped=s_, first_approached=s_, targets=targets,
                             primary=int(d["primary"][i, k]), target_vis=float(min(d["fvis"][i][t] for t in targets)),
                             policy=policy, mode="selection",
                             success=bool(correct), harvested=bool(correct), detached=s_ >= 0, target_correct=bool(correct),
                             wrong_target=bool(s_ >= 0 and not correct), approach_correct=bool(correct), touched_nontarget=False))
    return recs


def run(net, ck, data, mode="normal", dev="cuda", split_name="", bs=128, limit=0, raw=False, snap=DECODE_RADIUS):
    d = np.load(f"{data}/meta.npz")
    img = np.load(f"{data}/img.npy", mmap_mode="r") if ck.get("arch") != "symbolic" else None
    ok, ncmd = d["ok"].copy(), d["ncmd"]
    if limit > 0:
        ok[limit:] = False                                        # evaluate only the first `limit` scenes
    fxy, fvalid, tmask, tok, text = torch.as_tensor(d["fxy"]), torch.as_tensor(d["fvalid"]), d["tmask"], d["tok"], d["text"]
    N = len(ok)
    sel = np.full((N, 2), -2, np.int64)
    sym = ck.get("arch") == "symbolic"
    feats_all = fruit_features(fxy, torch.as_tensor(d["frad"]), torch.as_tensor(d["fmat"]), torch.as_tensor(d["fvis"]), fvalid) if sym else None
    blank = torch.as_tensor(tokenize(""))
    rng = np.random.RandomState(0)
    words = [w for w in VOCAB if not w.startswith("<")]
    with torch.no_grad():
        for s in range(0, N, bs):
            idx = np.arange(s, min(N, s + bs))
            x = feats_all[idx].to(dev) if sym else to_input(np.stack([img[i] for i in idx]), dev)
            for k in range(2):
                if k == 1 and not (ncmd[idx] > 1).any():
                    continue
                t = command_tokens(mode, ck, tok, idx, k, rng, words, blank, raw)
                lg = net(x, t.to(dev))
                sel[idx, k] = (lg.argmax(1) if sym else decode(lg, fxy[idx].to(dev), fvalid[idx].to(dev), radius=snap)).cpu().numpy()
    return make_records(d, sel, ok, ncmd, f"sel[{os.path.basename(ck.get('_path', ''))}|{mode}]", split_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="normal", choices=["normal", "blank", "gibberish", "swap"])
    ap.add_argument("--split", default="")
    ap.add_argument("--scenes", type=int, default=0)
    ap.add_argument("--raw", type=int, default=0, help="1: do not force the blank command on a blank-trained checkpoint (diagnostic)")
    ap.add_argument("--snap", type=float, default=DECODE_RADIUS, help="snap radius in px of the pixel-to-fruit rule (default 20)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = load_net(a.ckpt, dev)
    ck["_path"] = a.ckpt
    info = json.load(open(f"{a.data}/info.json"))
    recs = run(net, ck, a.data, a.mode, dev, a.split or info["split"], limit=a.scenes, raw=bool(a.raw), snap=a.snap)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    from lychee import evalkit
    s = evalkit.summarize(recs)["all"]
    print(f"{os.path.basename(a.ckpt)} on {a.split or info['split']} [{a.mode}]: {s['n_pairs']} pairs  PTA {evalkit.fmt(s['PTA_sel'])}  "
          f"TSA {evalkit.fmt(s['TSA'])}  collapse {evalkit.fmt(s['same_first_fruit'])}  chance {100*s['chance_PTA']:.1f}")
    for fam, m in evalkit.summarize(recs, by="family").items():
        print(f"    {fam:12s} n={m['n_pairs']:3d} PTA {evalkit.fmt(m['PTA_sel'])}")
