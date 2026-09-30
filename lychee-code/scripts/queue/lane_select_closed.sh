#!/bin/bash
# closed-loop check of selectors: the network chooses the fruit, the scripted expert executes (60 IID pairs, both commands)
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
for a in r1_film r0_rho90_film r0_rho00_film; do
  until [ -f $CK/$a.pt ]; do sleep 60; done
  out=results/eval/sel_${a}__iid.jsonl
  [ -s $out ] && continue
  acquire 9
  w=$(pick_workers)
  echo "== closed loop $a workers=$w $(date)"
  $PY -u scripts/eval_policy.py --policy "sel|$CK/$a.pt" --obs rgb --split iid --n 60 --workers $w --out $out 2>&1 | grep -v "^F:\|warnings.warn" | grep -v "^    "
done
echo "SELECT_CLOSED_DONE $(date)"
