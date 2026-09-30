#!/bin/bash
# out-of-distribution evaluation sets (occlusion, density, language, attribute): render as soon as the rho00 set is done, evaluate the arms that exist
source "$(dirname "$0")/common.sh"
until [ -f $DATA/select/rho00/info.json ]; do sleep 60; done
CK=$DATA/select_ckpt
for spec in "occ_ood occ_300 occ" "density_ood dens_300 dens" "lang_ood lang_300 lang" "attr_ood attr_300 attr"; do
  set -- $spec
  d=$DATA/select_eval/$2
  if [ ! -f $d/info.json ]; then
    acquire 7
    echo "== render $2 $(date)"
    $PY -u scripts/render_select.py --split $1 --start 0 --n 300 --mode pair --workers 3 --label_workers 8 --out $d 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
  fi
done
evo() {  # evaluate every existing arm on the four OOD sets (skips finished files)
  for a in "$@"; do
    [ -f $CK/$a.pt ] || continue
    for spec in "occ occ_300" "dens dens_300" "lang lang_300" "attr attr_300"; do
      set -- $spec
      [ -s results/eval/s_${a}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$a.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${a}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
    done
  done
}
ARMS="r1_film r0_rho00_film r0_rho90_film r0_rho50_film r1u_film r1_blank_film r1_late r1_token sym_r1 sym_r0_rho00 sym_r0_rho90 sym_r0_rho50 sym_r1u sym_r1_blank"
evo $ARMS          # early look with whatever is trained now
until grep -q "SELECT_TRAIN_DONE" results/lane_select_train.log 2>/dev/null && grep -q "SELECT_SYM_DONE" results/lane_select_sym.log 2>/dev/null; do sleep 120; done
evo $ARMS          # everything
echo "SELECT_OOD_DONE $(date)"
