#!/bin/bash
# Exploratory: does the bias dial hurt more when the command enters the network late?  R0 at rho = 0 and 0.9 with the two other conditionings
# (late fusion, command token); FiLM arms exist already.  OpenVLA-OFT motivates FiLM modulation with policies that ignore the instruction.
# Starts after the zero-contrast lane so that at most two heavy jobs share the laptop.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_SAME_DONE" results/lane_select_same.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
for cond in late token; do
  for r in 00 90; do
    name=r0_rho${r}_${cond}
    if [ ! -f $CK/$name.pt ]; then
      acquire 6
      echo "== train $name $(date)"
      $PYT -u scripts/train_select.py --data $DATA/select/rho$r --out $CK/$name.pt --arch pixel --cond $cond --epochs 30 > results/select_train_$name.log 2>&1
      tail -n 2 results/select_train_$name.log | head -n 1
    fi
    for spec in "iid iid_600" "sal sal_600" "attr attr_300"; do
      set -- $spec
      [ -f $CK/$name.pt ] || break
      [ -s results/eval/s_${name}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$name.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${name}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
    done
  done
done
echo "SELECT_COND_DONE $(date)"
