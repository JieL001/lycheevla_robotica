#!/bin/bash
# privileged symbolic reference selectors (fruit table instead of pixels) on the same datasets as the pixel arms
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
mkdir -p $CK results/eval
ready() { until [ -f $DATA/$1/info.json ]; do sleep 30; done; }
trn() {
  local name=$1 ds=$2; shift 2
  [ -f $CK/$name.pt ] && return 0
  ready $ds
  for attempt in 1 2 3; do
    acquire 5
    echo "== train $name (attempt $attempt) $(date)"
    $PYT -u scripts/train_select.py --data $DATA/$ds --out $CK/$name.pt --arch symbolic "$@" > results/select_train_$name.log 2>&1
    [ -f $CK/$name.pt ] && { tail -n 2 results/select_train_$name.log | head -n 1; return 0; }
    sleep 60
  done
}
evs() {
  local n=$1
  [ -f $CK/$n.pt ] || return 0
  for sp in "iid select_eval/iid_600" "sal select_eval/sal_600"; do
    set -- $sp
    [ -s results/eval/s_${n}__$1.jsonl ] || { ready $2; $PYT scripts/eval_select.py --ckpt $CK/$n.pt --data $DATA/$2 --split $1 --out results/eval/s_${n}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1; }
  done
  for mode in swap blank; do
    [ -s results/eval/s_${n}_${mode}__iid.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$n.pt --data $DATA/select_eval/iid_600 --split iid --mode $mode --out results/eval/s_${n}_${mode}__iid.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
  done
}
arm() { trn "$@"; evs $1; }
E=30
arm sym_r1       select/paired --epochs $E
arm sym_r0_rho00 select/rho00  --epochs $E
arm sym_r0_rho90 select/rho90  --epochs $E
arm sym_r1u      select/indep  --epochs $E
arm sym_r0_rho50 select/rho50  --epochs $E
arm sym_r1_blank select/paired --epochs $E --blank 1
echo "SELECT_SYM_DONE $(date)"
