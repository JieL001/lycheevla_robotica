#!/bin/bash
# Marginal-matched control for the bias dial (external review, 9/30): the scenes of the natural set with ONE command per scene drawn at the frequencies of the biased set (rho = 0.9) but not
# steered towards any fruit (scripts/make_matched_dataset.py -> $DATA/select/matched90).  FiLM, the recipe of the other R0 arms, 3 seeds; evaluated on the pilot and the fresh scenes.
# If it loses as much as R0 with rho = 0.9, the loss is a consequence of the mix of command types and not of the correlation with the cue.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
EV=$DATA/select_eval
SETS="iid:$EV/iid_600 sal:$EV/sal_600 attr:$EV/attr_300 iidf:$EV/iid_fresh_600 salf:$EV/sal_fresh_600 salb:$EV/sal_both_600 attrf:$EV/attr_fresh_300"
for s in 0 1 2; do
  name=r0_matched90_film; [ $s != 0 ] && name=${name}_s$s
  if [ ! -f $CK/$name.pt ]; then
    acquire 6
    echo "== train $name $(date)"
    $PYT -u scripts/train_select.py --data $DATA/select/matched90 --out $CK/$name.pt --arch pixel --cond film --epochs 30 --seed $s > results/select_train_$name.log 2>&1
    tail -n 2 results/select_train_$name.log | head -n 1
  fi
  [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
done
echo "MATCHED_DONE $(date)"
