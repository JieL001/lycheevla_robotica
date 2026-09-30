#!/bin/bash
# out-of-distribution and visual-jitter evaluation of the extra seeds (s1, s2) and of the dial arms, after the second wave has finished
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_EXTRA_DONE" results/lane_select_extra.log 2>/dev/null && grep -q "SELECT_DIAL_DONE" results/lane_select_dial.log 2>/dev/null; do sleep 120; done
CK=$DATA/select_ckpt
for a in r1_film_s1 r1_film_s2 r0_rho00_film_s1 r0_rho00_film_s2 r0_rho90_film_s1 r0_rho90_film_s2 r1u_film_s1 r1u_film_s2 r0_rho97_film r0_rho99_film sym_r0_rho97 sym_r0_rho99; do
  [ -f $CK/$a.pt ] || continue
  for spec in "iid iid_600" "sal sal_600" "occ occ_300" "dens dens_300" "attr attr_300" "dr dr_300"; do
    set -- $spec
    [ -s results/eval/s_${a}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$a.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${a}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
  done
done
echo "SELECT_SEEDOOD_DONE $(date)"
