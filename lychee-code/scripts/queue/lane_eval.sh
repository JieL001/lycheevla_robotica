#!/bin/bash
# evaluation lane: every trained arm, one evaluation at a time (3 simulator workers each).
#   iid n=60 (normal command), swap probe n=20; bias-dial arms and the R0/R1 references also on saliency_rev n=60
source "$(dirname "$0")/common.sh"
ARMS="r1_film r0_film r5_full r0_late r1_late r0_b90_film r1u_film r0_b50_film r1_film_blank r5_sup_only r5_inj_only r5_full_lang r0_film_n300 r0_film_n600"
SR_ARMS=" r1_film r0_film r0_b90_film r0_b50_film r1u_film r5_full r1_film_blank "
ev() {  # ev <arm> <mode> <split> <n> <out>
  local mode=$2; [ "$mode" = normal ] && spec="bc|$DATA/ckpt/$1.pt|4" || spec="bc|$DATA/ckpt/$1.pt|4|$mode"
  [ -s results/eval/$5 ] && [ "$(wc -l < results/eval/$5)" -ge $((2*$4)) ] && return 0
  acquire 9
  local w; w=$(pick_workers)
  echo "== eval $1 $mode $3 n=$4 workers=$w $(date)"
  $PY -u scripts/eval_policy.py --policy "$spec" --obs rgb --split $3 --n $4 --workers $w --out results/eval/$5 2>&1 | grep -v "^F:\|warnings.warn" | grep -v "^    "
}
while true; do
  pending=0
  for a in $ARMS; do
    if saved $a; then
      m=normal; [ "$a" = r1_film_blank ] && m=blank
      ev $a $m iid 60 bc_${a}__iid.jsonl
      case "$SR_ARMS" in *" $a "*) ev $a $m saliency_rev 60 bc_${a}__saliency_rev.jsonl;; esac
      [ "$a" != r1_film_blank ] && ev $a swap iid 20 bc_${a}_swap__iid.jsonl
    else
      pending=1
    fi
  done
  [ $pending -eq 0 ] && break
  sleep 60
done
echo "EVAL_LANE_DONE $(date)"
