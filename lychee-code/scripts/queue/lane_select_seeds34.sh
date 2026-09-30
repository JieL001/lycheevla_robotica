#!/bin/bash
# Seeds 3 and 4 of the four arms of the planned comparisons (R1, R0 at rho = 0 and 0.9, R1u): five seeds each instead of three, so that the seed-level intervals of C1-C3 and of the
# pairing and composition effects are not resting on two degrees of freedom.  Training changes the initialisation and the data order only.  Each arm is evaluated on every split
# except the language split (uninformative for encoders trained from scratch).
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
E=30
SETS="iid:$DATA/select_eval/iid_600 sal:$DATA/select_eval/sal_600 attr:$DATA/select_eval/attr_300 occ:$DATA/select_eval/occ_300 dens:$DATA/select_eval/dens_300 dr:$DATA/select_eval/dr_300"
for sd in 3 4; do
  for spec in "r1_film select/paired" "r0_rho00_film select/rho00" "r0_rho90_film select/rho90" "r1u_film select/indep"; do
    set -- $spec
    name=${1}_s$sd
    if [ ! -f $CK/$name.pt ]; then
      acquire 6
      echo "== train $name $(date)"
      $PYT -u scripts/train_select.py --data $DATA/$2 --out $CK/$name.pt --arch pixel --cond film --epochs $E --seed $sd > results/select_train_$name.log 2>&1
      tail -n 2 results/select_train_$name.log | head -n 1
    fi
    [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
  done
done
echo "SELECT_SEEDS34_DONE $(date)"
