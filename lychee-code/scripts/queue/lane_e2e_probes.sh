#!/bin/bash
# Language probes of the end-to-end policy (60 epochs, 40 IID scenes): the command of the other member of the pair (swap) and a blank command.
# NOTE (9/30): the swap run turned out to repeat the SAME (scene, command) episodes as the normal run with the two records of a pair exchanged (78 of 80 outcomes identical, the two others are
# the harvests judged against the other target set), so it is not an independent test and is not used in the paper; the paper uses the pair statistics of scripts/e2e_summary.py (the first
# fruit approached differs between the two commands; an approach under one command only) and the blank run.
source "$(dirname "$0")/common.sh"
CKPT=$DATA/ckpt/r1_film_long.pt
for mode in swap blank; do
  out=results/eval/bc_r1_film_long_${mode}__iid.jsonl
  [ -s $out ] && continue
  acquire 9
  w=$(pick_workers)
  echo "== e2e probe $mode workers=$w $(date)"
  $PY -u scripts/eval_policy.py --policy "bc|$CKPT|4|$mode" --obs rgb --split iid --n 40 --workers $w --out $out 2>&1 | grep -v "^F:\|warnings.warn" | grep -v "^    "
done
echo "E2E_PROBES_DONE $(date)"
