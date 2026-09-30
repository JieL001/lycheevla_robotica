#!/bin/bash
# Modular baseline without the label margins (9/30, after noticing that the strict benchmark rules turn small perception errors into commands without a target): the same detectors and parser,
# selection by the extremal / k-th detection (lychee.modular.choose_detection(strict=False)); tags mod_free (detector seeds 0-2) and mod_free_oracle (true fruit table).  Evaluation only.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
EV=$DATA/select_eval
SETS="iid:$EV/iid_600 sal:$EV/sal_600 occ:$EV/occ_300 dens:$EV/dens_300 lang:$EV/lang_300 attr:$EV/attr_300 dr:$EV/dr_300 iidf:$EV/iid_fresh_600 salf:$EV/sal_fresh_600 salb:$EV/sal_both_600 attrf:$EV/attr_fresh_300"
acquire 4
$PYT scripts/eval_modular.py --oracle 1 --strict 0 --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn" | cut -c1-170
$PYT scripts/eval_modular.py --strict 0 --tag mod_free --ckpt_dir $CK --ckpts detector_s0 detector_s1 detector_s2 --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn" | cut -c1-170
echo "MODFREE_DONE $(date)"
