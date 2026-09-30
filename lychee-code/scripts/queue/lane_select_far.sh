#!/bin/bash
# Coverage control for the bias dial: the natural set R0_0 with "farthest" commands cut to the number that the dial R0_0.9 leaves (146 of 8,000), everything else unchanged.
# If the loss on depth commands under the dial is a coverage effect, this arm shows it without any correlation between commands and salience.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_R0P_DONE" results/lane_select_r0p.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
name=r0_rho00_far146
if [ ! -f $CK/$name.pt ]; then
  acquire 6
  echo "== train $name $(date)"
  $PYT -u scripts/train_select.py --data $DATA/select/rho00 --out $CK/$name.pt --arch pixel --cond film --epochs 30 --far_keep 146 > results/select_train_$name.log 2>&1
  tail -n 2 results/select_train_$name.log | head -n 1
fi
for spec in "iid iid_600" "sal sal_600" "attr attr_300"; do
  set -- $spec
  [ -f $CK/$name.pt ] || break
  [ -s results/eval/s_${name}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$name.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${name}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
done
echo "SELECT_FAR_DONE $(date)"
