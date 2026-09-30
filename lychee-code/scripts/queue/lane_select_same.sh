#!/bin/bash
# Exploratory zero-contrast control R1n: the scenes of R1, and the second command is a paraphrase of the first (identical target set), so the
# contrast is removed completely. Added after the analysis showed that the independent-command control R1u still names disjoint fruit in
# 56% of its scenes. Together with R1u (56%) and R1 (100%) it gives a contrast dose of 0 / 56 / 100 %. Starts when the dial lane is done.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_DIAL_DONE" results/lane_select_dial.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
d=$DATA/select/same
if [ ! -f $d/info.json ]; then
  acquire 7
  echo "== render same $(date)"
  $PY -u scripts/render_select.py --split train --start 0 --n 4000 --mode same --workers 3 --label_workers 8 --out $d 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
fi
name=r1n_film
if [ ! -f $CK/$name.pt ]; then
  acquire 6
  echo "== train $name $(date)"
  $PYT -u scripts/train_select.py --data $d --out $CK/$name.pt --arch pixel --cond film --epochs 30 > results/select_train_$name.log 2>&1
  tail -n 2 results/select_train_$name.log | head -n 1
fi
for spec in "iid iid_600" "sal sal_600" "occ occ_300" "dens dens_300" "lang lang_300" "attr attr_300" "dr dr_300"; do
  set -- $spec
  [ -f $CK/$name.pt ] || break
  [ -s results/eval/s_${name}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$name.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${name}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
done
echo "SELECT_SAME_DONE $(date)"
