#!/bin/bash
# selection track: train each arm, then evaluate it offline (iid 600 scenes, saliency-reversed 600, language probes on iid)
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
mkdir -p $CK results/eval
ready() { until [ -f $DATA/$1/info.json ]; do sleep 30; done; }
trn() {  # trn <name> <dataset dir> <train_select.py flags...>
  local name=$1 ds=$2; shift 2
  [ -f $CK/$name.pt ] && return 0
  ready $ds
  for attempt in 1 2 3; do
    acquire 6
    echo "== train $name (attempt $attempt) $(date)"
    $PYT -u scripts/train_select.py --data $DATA/$ds --out $CK/$name.pt "$@" > results/select_train_$name.log 2>&1
    [ -f $CK/$name.pt ] && { tail -n 3 results/select_train_$name.log | head -n 2; return 0; }
    echo "   failed: $(tail -n 1 results/select_train_$name.log | cut -c1-160)"; sleep 60
  done
}
evs() {  # evs <name> : offline evaluation of a trained arm
  local n=$1
  [ -f $CK/$n.pt ] || return 0
  for sp in "iid select_eval/iid_600" "sal select_eval/sal_600"; do
    set -- $sp
    [ -s results/eval/s_${n}__$1.jsonl ] || { ready $2; $PYT scripts/eval_select.py --ckpt $CK/$n.pt --data $DATA/$2 --split $1 --out results/eval/s_${n}__$1.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1; }
  done
  for mode in swap blank gibberish; do
    [ -s results/eval/s_${n}_${mode}__iid.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/$n.pt --data $DATA/select_eval/iid_600 --split iid --mode $mode --out results/eval/s_${n}_${mode}__iid.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
  done
}
arm() { trn "$@"; evs $1; }
E=30
arm r1_film      select/paired --cond film  --epochs $E
arm r0_rho00_film select/rho00 --cond film  --epochs $E
arm r0_rho90_film select/rho90 --cond film  --epochs $E
arm r1u_film     select/indep  --cond film  --epochs $E
arm r0_rho50_film select/rho50 --cond film  --epochs $E
arm r1_blank_film select/paired --cond film --epochs $E --blank 1
arm r1_late      select/paired --cond late  --epochs $E
arm r1_token     select/paired --cond token --epochs $E
echo "SELECT_TRAIN_DONE $(date)"
