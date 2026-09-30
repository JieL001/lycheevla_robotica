#!/bin/bash
# Modular baseline: train the fruit detector (3 seeds, same recipe as the selection networks) and evaluate detector + rule parser on every evaluation set; also the oracle run
# (true fruit table through the same parser and rules).  Needs a free GPU; runs serially.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
EV=$DATA/select_eval
SETS="iid:$EV/iid_600 sal:$EV/sal_600 occ:$EV/occ_300 dens:$EV/dens_300 lang:$EV/lang_300 attr:$EV/attr_300 dr:$EV/dr_300 iidf:$EV/iid_fresh_600 salf:$EV/sal_fresh_600 salb:$EV/sal_both_600 attrf:$EV/attr_fresh_300"
$PYT scripts/eval_modular.py --oracle 1 --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
for s in 0 1 2; do
  [ -f $CK/detector_s$s.pt ] || { acquire 6; $PYT -u scripts/train_detector.py --data $DATA/select/rho00 --out $CK/detector_s$s.pt --epochs 30 --seed $s > results/select_train_detector_s$s.log 2>&1; }
done
$PYT scripts/eval_modular.py --ckpt_dir $CK --ckpts detector_s0 detector_s1 detector_s2 --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
echo "DETECTOR_DONE $(date)"
