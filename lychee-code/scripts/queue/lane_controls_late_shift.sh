#!/bin/bash
# Second review round (9/30 evening): the image-shift controls of Table 11 exist for FiLM only, so they cannot exclude that late fusion is affected by the label-misaligned shift in a way
# that FiLM is not.  Late fusion (0.55 M, the recipe of r1_late) without image shifts and with synchronised shifts, 3 seeds each.
#   bash scripts/queue/lane_controls_late_shift.sh none|sync
source "$(dirname "$0")/common.sh"
MODE=${1:?none|sync}
CK=$DATA/select_ckpt
EV=$DATA/select_eval
E=30
SETS="iid:$EV/iid_600 sal:$EV/sal_600 attr:$EV/attr_300 iidf:$EV/iid_fresh_600 salf:$EV/sal_fresh_600 attrf:$EV/attr_fresh_300"
for s in "" _s1 _s2; do
  name=r1_late_shift$MODE$s
  sd=0; case $name in *_s[0-9]) sd=${name##*_s};; esac
  if [ ! -f $CK/$name.pt ]; then
    acquire 4
    echo "== train $name $(date)"
    $PYT -u scripts/train_select.py --data $DATA/select/paired --out $CK/$name.pt --arch pixel --epochs $E --seed $sd --cond late --shift $MODE > results/select_train_$name.log 2>&1
    tail -n 2 results/select_train_$name.log | head -n 1
  fi
  [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
done
echo "LATE_SHIFT_${MODE}_DONE $(date)"
