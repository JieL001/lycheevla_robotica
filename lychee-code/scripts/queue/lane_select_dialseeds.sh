#!/bin/bash
# Seeds 1 and 2 of the other dial settings (rho = 0.5, 0.97, 0.99), so that the whole dial curve, and not only its two planned ends, rests on three seeds.
# Each arm is evaluated on the IID, saliency-reversed and attribute-OOD splits.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
E=30
SETS="iid:$DATA/select_eval/iid_600 sal:$DATA/select_eval/sal_600 attr:$DATA/select_eval/attr_300"
for sd in 1 2; do
  for r in 97 99 50; do
    name=r0_rho${r}_film_s$sd
    if [ ! -f $CK/$name.pt ]; then
      acquire 6
      echo "== train $name $(date)"
      $PYT -u scripts/train_select.py --data $DATA/select/rho$r --out $CK/$name.pt --arch pixel --cond film --epochs $E --seed $sd > results/select_train_$name.log 2>&1
      tail -n 2 results/select_train_$name.log | head -n 1
    fi
    [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
  done
done
echo "SELECT_DIALSEEDS_DONE $(date)"
