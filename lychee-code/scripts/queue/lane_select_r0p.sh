#!/bin/bash
# Exploratory: which property of the paired data gives the better composition on held-out (family, maturity) combinations (R1, R1u and even R1n beat R0 there)?
# R0p = one command per scene on the scenes of the paired set (the first command of every pair, no second command), the same number of updates as R1.
# Compared with R1n (same scenes, two wordings of that command) it isolates the repeated wording; compared with R0 on 4,000 unpaired scenes it isolates the
# scene / command distribution of the pair generator.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
name=r0p_film
if [ ! -f $CK/$name.pt ]; then
  acquire 6
  echo "== train $name $(date)"
  $PYT -u scripts/train_select.py --data $DATA/select/paired --out $CK/$name.pt --arch pixel --cond film --cmd0_only 1 --total_steps 3500 > results/select_train_$name.log 2>&1
  tail -n 2 results/select_train_$name.log | head -n 1
fi
for spec in "iid iid_600" "sal sal_600" "attr attr_300" "occ occ_300" "dens dens_300" "lang lang_300" "dr dr_300"; do
  set -- $spec
  [ -f $CK/$name.pt ] || break
  [ -s results/eval/s_${name}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$name.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${name}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
done
echo "SELECT_R0P_DONE $(date)"
