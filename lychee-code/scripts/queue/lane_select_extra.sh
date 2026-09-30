#!/bin/bash
# selection track, second wave: more seeds for the confirmatory arms, data scaling, scene-binding dial, symbolic seeds.
# Starts only after the first wave (lane_select_train.sh) has finished, so that the GPU and the free commit are not shared.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_TRAIN_DONE" results/lane_select_train.log 2>/dev/null; do sleep 60; done
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
# seeds 1 and 2 of the arms in C1-C3 (pixel selectors, film conditioning)
for sd in 1 2; do
  for spec in "r1_film select/paired" "r0_rho00_film select/rho00" "r0_rho90_film select/rho90" "r1u_film select/indep"; do
    set -- $spec
    trn ${1}_s$sd $2 pixel --cond film --epochs $E --seed $sd; evn ${1}_s$sd
  done
done
# data scaling of R1 and R0 rho=0 (number of labelled samples)
for n in 1000 2000 4000; do
  trn r1_film_n$n select/paired pixel --cond film --epochs $E --max_samples $n; evn r1_film_n$n
  trn r0_rho00_film_n$n select/rho00 pixel --cond film --epochs $E --max_samples $n; evn r0_rho00_film_n$n
done
# scene-binding dial: K training scenes; one command per scene (bind_r0) versus both commands (bind_r1); same number of updates
for K in 100 1000; do
  for arm in bind_r0 bind_r1; do
    fl="--total_steps 3000"; [ $arm = bind_r0 ] && fl="--cmd0_only 1 --total_steps 3000"
    trn ${arm}_K$K select/paired pixel --cond film --scenes $K $fl
    if [ -f $CK/${arm}_K$K.pt ]; then
      # counterfactual test on the training scenes themselves: both commands of every one of the K scenes
      [ -s results/eval/s_${arm}_K${K}__cf.jsonl ] || $PYT scripts/eval_select.py --ckpt $CK/${arm}_K$K.pt --data $DATA/select/paired --scenes $K --split cf --out results/eval/s_${arm}_K${K}__cf.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | head -n 1
      evn ${arm}_K$K
    fi
  done
done
echo "SELECT_EXTRA_DONE $(date)"
