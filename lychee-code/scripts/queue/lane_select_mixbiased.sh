#!/bin/bash
# Biased-plus-biased control of the remedy experiment: the biased set R0_0.9 (8,000 scenes, one command each) plus 4,000 MORE biased scenes (rho = 0.9, new scene indices), the same
# number of samples and updates as the two remedy arms, so that "unbiased data restore PTA" can be told from "more training restores PTA".
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
S=$DATA/select
SETS="iid:$DATA/select_eval/iid_600 sal:$DATA/select_eval/sal_600 attr:$DATA/select_eval/attr_300"
if [ ! -f $S/rho90_extra/info.json ]; then
  acquire 7
  echo "== render rho90_extra $(date)"
  $PY -u scripts/render_select.py --split train --rho 0.9 --start 8000 --n 4000 --mode single --workers 3 --label_workers 8 --out $S/rho90_extra 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
fi
[ -f $S/mix_rho90_biased/info.json ] || $PY scripts/make_mix_dataset.py --a $S/rho90 --b $S/rho90_extra --nb 4000 --out $S/mix_rho90_biased
name=mix_rho90_biased
if [ ! -f $CK/$name.pt ]; then
  acquire 6
  echo "== train $name $(date)"
  $PYT -u scripts/train_select.py --data $S/$name --out $CK/$name.pt --arch pixel --cond film --epochs 30 > results/select_train_$name.log 2>&1
  tail -n 2 results/select_train_$name.log | head -n 1
fi
[ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
echo "SELECT_MIXBIASED_DONE $(date)"
