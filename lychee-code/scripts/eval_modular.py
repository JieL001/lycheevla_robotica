"""Evaluate the modular baseline (fruit detector + rule-based parser + the benchmark's rules on the predicted fruit table) on rendered evaluation sets, selection level.
   python scripts/eval_modular.py --ckpt_dir D:/lychee_data/select_ckpt --ckpts detector_s0 detector_s1 \
       --sets iid:D:/lychee_data/select_eval/iid_600 sal:D:/lychee_data/select_eval/sal_600 [--snap 20] [--tag mod_det] [--outdir results/eval]
   python scripts/eval_modular.py --oracle 1 --sets ...        # perfect perception: the true fruit table through the same parser and rules (validates parser and rules against the labels)
Writes results/eval/s_<tag>[_s1|_s2]__<split>.jsonl with the fields of scripts/eval_select.py (the checkpoint detector_s<k> gives <tag> for k = 0 and <tag>_s<k> otherwise), so that
evalkit.summarize and the table scripts treat it like a selection network.  Also prints the detection quality on each set (matching radius MATCH_PX).  The pointing position is
snapped to the nearest true fruit centre within --snap px (default 20), as for the learned selectors."""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import eval_select as es
from lychee import evalkit
from lychee.detector import FruitDetector, decode_detections, detections_from_table, match_detections
from lychee.modular import choose_detection, parse_command, snap
from lychee.select import DECODE_RADIUS, to_input


def run(net, thr, data, split, snap_px=DECODE_RADIUS, oracle=False, dev="cuda", bs=128, policy="mod", strict=True):
    d = np.load(f"{data}/meta.npz")
    ok, ncmd = d["ok"], d["ncmd"]
    img = None if oracle else np.load(f"{data}/img.npy", mmap_mode="r")
    N = len(ok)
    sel = np.full((N, 2), -2, np.int64)
    tp = nd = ng = mat_ok = 0
    parsed = total = 0
    errs = {}                                                   # family -> [commands, rules returned no target, pointing position too far from every fruit, wrong fruit]
    for s in range(0, N, bs):
        idx = np.arange(s, min(N, s + bs))
        if oracle:
            dets = [detections_from_table(d["fxy"][i], d["frad"][i], d["fmat"][i], d["fvis"][i], d["fvalid"][i]) for i in idx]
        else:
            with torch.no_grad():
                dets = decode_detections(net(to_input(np.stack([img[i] for i in idx]), dev)), thr)
        for det, i in zip(dets, idx):
            if not ok[i]:
                continue
            pairs, n_det, n_gt = match_detections(det, d["fxy"][i], d["fvis"][i], d["fvalid"][i])
            tp += len(pairs); nd += n_det; ng += n_gt
            mat_ok += sum(int(det["mat"][j] == d["fmat"][i][f]) for j, f in pairs)
            for k in range(int(ncmd[i])):
                spec = parse_command(d["text"][i, k])
                total += 1; parsed += spec is not None
                j = choose_detection(det, spec, strict)
                sel[i, k] = snap(det, j, d["fxy"][i], d["fvalid"][i], snap_px)
                e = errs.setdefault(es.FAMS[int(d["family"][i])], [0, 0, 0, 0])
                e[0] += 1
                if j is None:
                    e[1] += 1
                elif sel[i, k] < 0:
                    e[2] += 1
                elif not d["tmask"][i, k, sel[i, k]]:
                    e[3] += 1
    recs = es.make_records(d, sel, ok, ncmd, policy, split)
    p, r = tp / max(1, nd), tp / max(1, ng)
    stats = dict(precision=p, recall=r, f1=2 * p * r / max(1e-9, p + r), maturity_acc=mat_ok / max(1, tp), parsed=parsed / max(1, total), errors=errs)
    return recs, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="D:/lychee_data/select_ckpt")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--sets", nargs="+", required=True, help="split:data_dir")
    ap.add_argument("--outdir", default="results/eval")
    ap.add_argument("--tag", default="mod_det")
    ap.add_argument("--snap", type=float, default=DECODE_RADIUS)
    ap.add_argument("--oracle", type=int, default=0)
    ap.add_argument("--strict", type=int, default=1, help="1: the benchmark's rules with their margins; 0: the same rules without the margins (tags mod_free, mod_free_oracle)")
    ap.add_argument("--suffix", default="", help="appended to the file name before __<split> (e.g. _snap10)")
    ap.add_argument("--force", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [("mod_oracle" if a.strict else "mod_free_oracle", None, None)] if a.oracle else []
    for name in ([] if a.oracle else a.ckpts):
        k = name.rsplit("_s", 1)[-1] if "_s" in name else "0"
        tag = a.tag if k == "0" else f"{a.tag}_s{k}"
        jobs.append((tag, name, k))
    for tag, name, _ in jobs:
        net = thr = None
        if name is not None:
            ck = torch.load(f"{a.ckpt_dir}/{name}.pt", map_location="cpu", weights_only=False)
            net = FruitDetector().to(dev); net.load_state_dict(ck["state"]); net.eval(); thr = ck["thr"]
        for sp, data in (s.split(":", 1) for s in a.sets):
            out = f"{a.outdir}/s_{tag}{a.suffix}__{sp}.jsonl"
            if not a.force and os.path.exists(out) and os.path.getsize(out) > 0:
                continue
            recs, st = run(net, thr, data, sp, a.snap, oracle=bool(a.oracle), dev=dev, policy=f"mod[{tag}]", strict=bool(a.strict))
            with open(out, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            js = f"{a.outdir}/modular_stats.json"                       # detection quality and error breakdown of every run (merged)
            allst = json.load(open(js)) if os.path.exists(js) else {}
            allst[f"{tag}{a.suffix}__{sp}"] = st
            json.dump(allst, open(js, "w"), indent=1)
            s = evalkit.summarize(recs)["all"]
            print(f"{tag}{a.suffix} on {sp}: {s['n_pairs']} pairs  PTA {evalkit.fmt(s['PTA_sel'])}  TSA {evalkit.fmt(s['TSA'])}  chance {100*s['chance_PTA']:.1f} | "
                  f"detector precision {st['precision']:.3f} recall {st['recall']:.3f} maturity acc {st['maturity_acc']:.3f} parsed {st['parsed']:.3f}", flush=True)
