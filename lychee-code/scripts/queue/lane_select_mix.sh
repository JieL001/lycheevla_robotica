#!/bin/bash
# Exploratory remedy-under-bias experiment (the experiment that planned comparison C2 was meant to be): the biased set R0_0.9 (8,000 scenes, one command each) plus
# 2,000 extra PAIRED scenes (4,000 samples) versus the same biased set plus 4,000 extra unpaired NATURAL scenes (4,000 samples). Same total samples and updates.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_R0P_DONE" results/lane_select_r0p.log 2>/dev/null; do sleep 60; done
CK=$DATA/select_ckpt
S=$DATA/select
[ -f $S/mix_rho90_paired/info.json ]  || $PY scripts/make_mix_dataset.py --a $S/rho90 --b $S/paired --nb 2000 --out $S/mix_rho90_paired
[ -f $S/mix_rho90_natural/info.json ] || $PY scripts/make_mix_dataset.py --a $S/rho90 --b $S/rho00 --nb 4000 --out $S/mix_rho90_natural
for name in mix_rho90_paired mix_rho90_natural; do
  if [ ! -f $CK/$name.pt ]; then
    acquire 6
    echo "== train $name $(date)"
    $PYT -u scripts/train_select.py --data $S/$name --out $CK/$name.pt --arch pixel --cond film --epochs 30 > results/select_train_$name.log 2>&1
    tail -n 2 results/select_train_$name.log | head -n 1
  fi
  for spec in "iid iid_600" "sal sal_600" "attr attr_300"; do
    set -- $spec
    [ -f $CK/$name.pt ] || break
    [ -s results/eval/s_${name}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$name.pt --data $DATA/select_eval/$2 --split $1 --out results/eval/s_${name}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
  done
done
echo "SELECT_MIX_DONE $(date)"
