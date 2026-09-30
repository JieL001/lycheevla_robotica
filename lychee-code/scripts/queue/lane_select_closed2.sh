#!/bin/bash
# closed-loop agreement on the shifted splits: the network chooses the fruit, the scripted expert executes (60 pairs, both commands) for R1 and R0_0.9 on the
# saliency-reversed, occlusion, density and attribute splits (the IID check exists). Needs the simulator, so it waits for free commit and runs when the trainers leave room.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
for spec in "sal saliency_rev" "occ occ_ood" "dens density_ood" "attr attr_ood"; do
  set -- $spec
  for a in r1_film r0_rho90_film; do
    out=results/eval/sel_${a}__$1.jsonl
    [ -s $out ] && continue
    acquire 9
    w=$(pick_workers)
    echo "== closed loop $a on $2 workers=$w $(date)"
    $PY -u scripts/eval_policy.py --policy "sel|$CK/$a.pt" --obs rgb --split $2 --n 60 --workers $w --out $out 2>&1 | grep -v "^F:\|warnings.warn" | grep -v "^    "
  done
done
echo "SELECT_CLOSED2_DONE $(date)"
