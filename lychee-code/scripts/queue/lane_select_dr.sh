#!/bin/bash
# robustness to held-out camera / lighting / colour jitter (visual_dr_ood): render 300 scenes, then evaluate the main arms
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_DATA_DONE" results/lane_select_data.log 2>/dev/null; do sleep 60; done
d=$DATA/select_eval/dr_300
if [ ! -f $d/info.json ]; then
  acquire 7
  echo "== render dr_300 $(date)"
  $PY -u scripts/render_select.py --split visual_dr_ood --start 0 --n 300 --mode pair --workers 3 --label_workers 8 --out $d 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
fi
until grep -q "SELECT_TRAIN_DONE" results/lane_select_train.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
for a in r1_film r0_rho00_film r0_rho90_film r1u_film r1_blank_film sym_r1 sym_r0_rho00 sym_r0_rho90; do
  [ -f $CK/$a.pt ] || continue
  [ -s results/eval/s_${a}__dr.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$a.pt --data $d --split dr --out results/eval/s_${a}__dr.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
done
echo "SELECT_DR_DONE $(date)"
