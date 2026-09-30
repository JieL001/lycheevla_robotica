#!/bin/bash
# Second seed of the two exploratory controls whose held-out-combination results are the most interesting (R1n: two wordings, no contrast; R0p: one command on the scenes of the
# pair generator), so that "two commands per scene help composition" does not rest on a single seed.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_R0P_DONE" results/lane_select_r0p.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
S=$DATA/select
trn() {  # trn <name> <flags...>
  local name=$1; shift
  [ -f $CK/$name.pt ] && return 0
  acquire 6
  echo "== train $name $(date)"
  $PYT -u scripts/train_select.py --out $CK/$name.pt --arch pixel --cond film "$@" > results/select_train_$name.log 2>&1
  tail -n 2 results/select_train_$name.log | head -n 1
}
ev() {
  local n=$1
  for spec in "iid iid_600" "sal sal_600" "attr attr_300" "occ occ_300" "dens dens_300"; do
    set -- $spec
    [ -f $CK/$n.pt ] || return 0
    [ -s results/eval/s_${n}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$n.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${n}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
  done
}
trn r1n_film_s1 --data $S/same --epochs 30 --seed 1; ev r1n_film_s1
trn r0p_film_s1 --data $S/paired --cmd0_only 1 --total_steps 3500 --seed 1; ev r0p_film_s1
echo "SELECT_CTLSEEDS_DONE $(date)"
