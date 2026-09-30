#!/bin/bash
# Data-scaling arms that the second-wave lanes were stopped before (the laptop ran out of memory with five concurrent jobs): R1 and R0 with 1,000 labelled samples, R1 with 2,000.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
trn() {  # trn <name> <dataset> <arch> <flags...>
  local name=$1 ds=$2 arch=$3; shift 3
  [ -f $CK/$name.pt ] && return 0
  acquire 6
  echo "== train $name $(date)"
  $PYT -u scripts/train_select.py --data $DATA/$ds --out $CK/$name.pt --arch $arch "$@" > results/select_train_$name.log 2>&1
  tail -n 2 results/select_train_$name.log | head -n 1
}
evn() {  # evn <name> : iid + saliency-reversed offline evaluation
  local n=$1
  [ -f $CK/$n.pt ] || return 0
  for sp in "iid select_eval/iid_600" "sal select_eval/sal_600"; do
    set -- $sp
    [ -s results/eval/s_${n}__$1.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$n.pt --data $DATA/$2 --split $1 --out results/eval/s_${n}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
  done
}
E=30
trn r1_film_n1000 select/paired pixel --cond film --epochs $E --max_samples 1000; evn r1_film_n1000
trn r0_rho00_film_n1000 select/rho00 pixel --cond film --epochs $E --max_samples 1000; evn r0_rho00_film_n1000
trn r1_film_n2000 select/paired pixel --cond film --epochs $E --max_samples 2000; evn r1_film_n2000
echo "SELECT_NARMS_DONE $(date)"
