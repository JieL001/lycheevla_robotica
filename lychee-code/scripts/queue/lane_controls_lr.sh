#!/bin/bash
# Second lane of the controls of the review (9/30): the four learning-rate runs (FiLM and late fusion at 3e-4 and 3e-3, seed 0, nothing selected), run next to lane_controls.sh.
# Same recipe as lane_controls.sh; needs less free commit than the other lanes because a training job next to it has already been running for a while.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
EV=$DATA/select_eval
E=30
SETS="iid:$EV/iid_600 sal:$EV/sal_600 attr:$EV/attr_300 iidf:$EV/iid_fresh_600 salf:$EV/sal_fresh_600 attrf:$EV/attr_fresh_300"
run() {  # run <name> <extra train_select.py flags...>
  local name=$1; shift
  local sd=0; case $name in *_s[0-9]) sd=${name##*_s};; esac
  if [ ! -f $CK/$name.pt ]; then
    acquire 4
    echo "== train $name $(date)"
    $PYT -u scripts/train_select.py --data $DATA/select/paired --out $CK/$name.pt --arch pixel --epochs $E --seed $sd "$@" > results/select_train_$name.log 2>&1
    tail -n 2 results/select_train_$name.log | head -n 1
  fi
  [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
}
run r1_film_lr3e-4 --cond film --lr 3e-4
run r1_film_lr3e-3 --cond film --lr 3e-3
run r1_late_lr3e-4 --cond late --lr 3e-4
run r1_late_lr3e-3 --cond late --lr 3e-3
echo "CONTROLS_LR_DONE $(date)"
