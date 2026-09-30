#!/bin/bash
# extends the bias dial to rho = 0.97 and 0.99 (pixel and fruit-table selectors); waits for the main render lane
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_DATA_DONE" results/lane_select_data.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
for r in 97 99; do
  d=$DATA/select/rho$r
  if [ ! -f $d/info.json ]; then
    acquire 7
    echo "== render rho$r $(date)"
    $PY -u scripts/render_select.py --split train --rho 0.$r --start 0 --n 8000 --mode single --workers 3 --label_workers 8 --out $d 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
  fi
done
until grep -q "SELECT_TRAIN_DONE" results/lane_select_train.log 2>/dev/null; do sleep 60; done
for r in 97 99; do
  for arch in pixel symbolic; do
    if [ $arch = pixel ]; then name=r0_rho${r}_film; flags="--cond film"; else name=sym_r0_rho$r; flags=""; fi
    if [ ! -f $CK/$name.pt ]; then
      acquire 6
      echo "== train $name $(date)"
      $PYT -u scripts/train_select.py --data $DATA/select/rho$r --out $CK/$name.pt --arch $arch $flags --epochs 30 > results/select_train_$name.log 2>&1
      tail -n 2 results/select_train_$name.log | head -n 1
    fi
    for sp in "iid select_eval/iid_600" "sal select_eval/sal_600"; do
      set -- $sp
      [ -s results/eval/s_${name}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$name.pt --data $DATA/$2 --split $1 --out results/eval/s_${name}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
    done
  done
done
echo "SELECT_DIAL_DONE $(date)"
