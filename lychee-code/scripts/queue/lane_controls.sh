#!/bin/bash
# Controls asked for by an external review (9/30): (1) image shifts: the reported arms shift the image by up to 4 px WITHOUT shifting the labels (--shift unsync); R1 with no shifts
# and with synchronised shifts (3 seeds each); (2) capacity: R1 with late fusion made as large as the FiLM variant (late_dim 256, 0.60 M against 0.61 M parameters; 3 seeds);
# (3) learning rate: FiLM and late fusion at 3e-4 and 3e-3 (seed 0, no selection among them: all are reported).  Waits for the detector lane so that at most one GPU job runs.
source "$(dirname "$0")/common.sh"
until grep -q "DETECTOR_DONE" results/lane_detector.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
EV=$DATA/select_eval
E=30
SETS="iid:$EV/iid_600 sal:$EV/sal_600 attr:$EV/attr_300 iidf:$EV/iid_fresh_600 salf:$EV/sal_fresh_600 attrf:$EV/attr_fresh_300"
run() {  # run <name> <extra train_select.py flags...>
  local name=$1; shift
  local sd=0; case $name in *_s[0-9]) sd=${name##*_s};; esac
  if [ ! -f $CK/$name.pt ]; then
    acquire 6
    echo "== train $name $(date)"
    $PYT -u scripts/train_select.py --data $DATA/select/paired --out $CK/$name.pt --arch pixel --epochs $E --seed $sd "$@" > results/select_train_$name.log 2>&1
    tail -n 2 results/select_train_$name.log | head -n 1
  fi
  [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
}
for s in "" _s1 _s2; do
  run r1_film_shiftnone$s --cond film --shift none
  run r1_film_shiftsync$s --cond film --shift sync
  run r1_late_wide$s --cond late --late_dim 256
done
run r1_film_lr3e-4 --cond film --lr 3e-4
run r1_film_lr3e-3 --cond film --lr 3e-3
run r1_late_lr3e-4 --cond late --lr 3e-4
run r1_late_lr3e-3 --cond late --lr 3e-3
echo "CONTROLS_DONE $(date)"
