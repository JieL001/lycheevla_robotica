"""Level 2 of REPRODUCE.md: re-render an evaluation set from its split definition and seed range, evaluate a shipped checkpoint on it and compare with the shipped records.
   python scripts/verify_evaluation.py --ckpt ../checkpoints/r1_film.pt --arm r1_film --split iid --data ./scratch/iid_600     [--device cpu]
The scenes are re-created by scripts/render_select.py (ManiSkill 3, CPU rendering, roughly 4-8 scenes per second; skipped if --data already holds a rendered set) and the checkpoint is
evaluated by scripts/eval_select.py at the default snap radius.  The script compares, for every (scene, command) of results/eval/s_<arm>__<split>.jsonl,
  (i) the command text and the target set (they depend on the scene only: any difference means that the scene was not re-created identically), and
  (ii) the selected fruit (first_detached); the selection of a network can flip on a handful of commands if the pixels differ in the last bits (other CPU, other torch build),
and prints PTA of the shipped records and of the re-evaluation.  Exit code 0 if all scenes and commands are identical and PTA differs by at most 1 point."""
import argparse, json, os, subprocess, sys

import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import eval_select as es
from lychee import evalkit

# split name of the shipped records -> (split name of lychee/splits.py, number of scenes); all sets are rendered as counterfactual pairs from scene index 0
SETS = {"iid": ("iid", 600), "sal": ("saliency_rev", 600), "occ": ("occ_ood", 300), "dens": ("density_ood", 300), "lang": ("lang_ood", 300), "attr": ("attr_ood", 300),
        "dr": ("visual_dr_ood", 300), "iidf": ("iid_fresh", 600), "salf": ("sal_fresh", 600), "salb": ("sal_both", 600), "attrf": ("attr_fresh", 300)}


def load(path):
    return {(r["index"], r["which"]): r for r in map(json.loads, open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint (.pt) of the arm, e.g. ../checkpoints/r1_film.pt")
    ap.add_argument("--arm", required=True, help="name of the records of this arm: results/eval/s_<arm>__<split>.jsonl (e.g. r1_film, r0_rho90_film_s1)")
    ap.add_argument("--split", required=True, choices=sorted(SETS))
    ap.add_argument("--data", required=True, help="directory of the rendered evaluation set (created if it does not exist)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", default="3", help="simulator workers of render_select.py (about 2 GB of memory each)")
    ap.add_argument("--label_workers", default="4", help="light worker processes that sample the commands and compute the labels (about 0.3 GB each)")
    a = ap.parse_args()
    rec_path = f"results/eval/s_{a.arm}__{a.split}.jsonl"
    if not os.path.exists(rec_path):
        sys.exit(f"shipped records not found: {rec_path}")
    split, n = SETS[a.split]
    if not os.path.exists(f"{a.data}/info.json"):
        print(f"rendering {n} scenes of split {split} into {a.data} ...", flush=True)
        subprocess.run([sys.executable, "-u", "scripts/render_select.py", "--split", split, "--start", "0", "--n", str(n), "--mode", "pair", "--workers", a.workers,
                        "--label_workers", a.label_workers, "--out", a.data], check=True)
    net, ck = es.load_net(a.ckpt, a.device)
    ck["_path"] = a.ckpt
    recs = es.run(net, ck, a.data, "normal", a.device, a.split)
    new = {(r["index"], r["which"]): r for r in recs}
    old = load(rec_path)
    keys_ok = set(new) == set(old)
    same_scene = sum(new[k]["text"] == old[k]["text"] and new[k]["targets"] == old[k]["targets"] for k in set(new) & set(old))
    same_sel = sum(new[k]["first_detached"] == old[k]["first_detached"] for k in set(new) & set(old))
    pta = lambda rs: evalkit.summarize(list(rs.values()))["all"]["PTA_sel"]
    p_old, p_new = pta(old), pta(new)
    tot = len(set(new) & set(old))
    print(f"{a.arm} on {a.split}: {len(old)} shipped records, {len(new)} re-evaluated, same (scene, command) keys: {keys_ok}")
    print(f"  identical command text and target set: {same_scene}/{tot}; identical selected fruit: {same_sel}/{tot}")
    print(f"  PTA shipped {evalkit.fmt(p_old)}   re-evaluated {evalkit.fmt(p_new)}")
    ok = keys_ok and same_scene == tot and abs(p_old[0] - p_new[0]) <= 0.01            # PTA is a (value, low, high) triple
    print("REPRODUCED" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
