"""Evaluate several selection networks on several rendered evaluation sets in ONE process (the start-up of Python and CUDA costs more than the evaluation of 600 scenes).
   python scripts/eval_select_batch.py --ckpt_dir D:/lychee_data/select_ckpt --ckpts r1_film r0_rho00_film_s1 \
       --sets iid:D:/lychee_data/select_eval/iid_600 sal:D:/lychee_data/select_eval/sal_600 [--outdir results/eval] [--force 0]
Writes results/eval/s_<ckpt>__<split>.jsonl exactly as scripts/eval_select.py does (same code path) and skips files that exist, so that it can be re-run after new checkpoints appear;
checkpoints that do not exist are skipped."""
import argparse, json, os, sys

import torch

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import eval_select as es
from lychee import evalkit

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="D:/lychee_data/select_ckpt")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="split:data_dir")
    ap.add_argument("--outdir", default="results/eval")
    ap.add_argument("--force", type=int, default=0)
    ap.add_argument("--snap", type=float, default=20.0, help="snap radius (px) of the pixel-to-fruit rule; 20 = the value of all reported results")
    ap.add_argument("--suffix", default="", help="appended to the file name before __<split>, e.g. _snap10 (sensitivity of the results to the snap radius)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in a.ckpts:
        path = f"{a.ckpt_dir}/{name}.pt"
        todo = [(sp, d) for sp, d in (s.split(":", 1) for s in a.sets)
                if a.force or not (os.path.exists(f"{a.outdir}/s_{name}{a.suffix}__{sp}.jsonl") and os.path.getsize(f"{a.outdir}/s_{name}{a.suffix}__{sp}.jsonl") > 0)]
        if not os.path.exists(path) or not todo:
            continue
        net, ck = es.load_net(path, dev)
        ck["_path"] = path
        for sp, data in todo:
            recs = es.run(net, ck, data, "normal", dev, sp, snap=a.snap)
            out = f"{a.outdir}/s_{name}{a.suffix}__{sp}.jsonl"
            with open(out, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            s = evalkit.summarize(recs)["all"]
            print(f"{name}.pt on {sp} [normal, snap {a.snap:g}]: {s['n_pairs']} pairs  PTA {evalkit.fmt(s['PTA_sel'])}  TSA {evalkit.fmt(s['TSA'])}  "
                  f"collapse {evalkit.fmt(s['same_first_fruit'])}  chance {100*s['chance_PTA']:.1f}", flush=True)
