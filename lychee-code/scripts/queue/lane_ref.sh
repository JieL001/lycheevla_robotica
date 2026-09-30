#!/bin/bash
# reference-policy lane: scripted policies on every split (2 simulator workers, gated on free commit). Skips finished files.
source "$(dirname "$0")/common.sh"
ev_ref() {  # ev_ref <policy> <split> <n>
  local out=results/eval/$1_$2.jsonl
  [ -s $out ] && [ "$(wc -l < $out)" -ge $((2*$3)) ] && return 0
  acquire 7
  echo "== ref $1 $2 n=$3 $(date)"
  $PY -u scripts/eval_policy.py --policy $1 --split $2 --n $3 --workers 2 --out $out 2>&1 | grep -v "^F:\|warnings.warn" | grep -v "^    "
}
ev_ref rel_only iid 60
for sp in occ_ood density_ood lang_ood attr_ood saliency_rev; do
  for pol in expert attr_visible rel_only blind_visible; do ev_ref $pol $sp 30; done
done
echo "REF_LANE_DONE $(date)"
