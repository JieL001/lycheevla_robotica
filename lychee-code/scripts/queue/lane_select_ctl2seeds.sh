#!/bin/bash
# Seeds 1 and 2 of the single-seed arms whose numbers carry statements of the paper (a mock review asked for them): the coverage control (R0 with 146 "farthest" commands), the two
# remedy-under-bias arms (biased set + paired scenes / + natural scenes) and R1 with late fusion and with a command token.  Evaluated on IID, saliency-reversed and attribute-OOD scenes.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
E=30
SETS="iid:$DATA/select_eval/iid_600 sal:$DATA/select_eval/sal_600 attr:$DATA/select_eval/attr_300"
run() {  # run <name> <dataset under $DATA> <train_select.py flags...>
  local name=$1 data=$2; shift 2
  if [ ! -f $CK/$name.pt ]; then
    acquire 6
    echo "== train $name $(date)"
    $PYT -u scripts/train_select.py --data $DATA/$data --out $CK/$name.pt --arch pixel --epochs $E "$@" > results/select_train_$name.log 2>&1
    tail -n 2 results/select_train_$name.log | head -n 1
  fi
  [ -f $CK/$name.pt ] && $PYT scripts/eval_select_batch.py --ckpts $name --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
}
for sd in 1 2; do
  run r0_rho00_far146_s$sd select/rho00 --cond film --far_keep 146 --seed $sd
  run r1_late_s$sd  select/paired --cond late  --seed $sd
  run r1_token_s$sd select/paired --cond token --seed $sd
  run mix_rho90_paired_s$sd  select/mix_rho90_paired  --cond film --seed $sd
  run mix_rho90_natural_s$sd select/mix_rho90_natural --cond film --seed $sd
done
echo "SELECT_CTL2SEEDS_DONE $(date)"
