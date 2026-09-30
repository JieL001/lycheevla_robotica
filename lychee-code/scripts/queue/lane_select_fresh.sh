#!/bin/bash
# Re-evaluation on FRESH scenes (rendered after the pilot; no design decision or arm choice has seen them): IID, saliency-reversed and attribute-OOD
# scenes with new seeds, and a saliency-reversed split in which BOTH commands of a pair avoid the most salient fruit (sal_both; the original split
# constrains only the first member).  Rendering starts when the extra-seed lane has finished, evaluation when every training lane has.
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_EXTRA_DONE" results/lane_select_extra.log 2>/dev/null; do sleep 120; done
rd() {  # rd <dir under $DATA> <render_select.py args...>
  local out=$1; shift
  [ -f $DATA/$out/info.json ] && return 0
  acquire 7
  echo "== render $out $(date)"
  $PY -u scripts/render_select.py "$@" --workers 3 --label_workers 8 --out $DATA/$out 2>&1 | grep -v "^F:\|warnings.warn" | grep --line-buffered -v "^    "
}
rd select_eval/iid_fresh_600  --split iid_fresh  --start 0 --n 600 --mode pair
rd select_eval/sal_fresh_600  --split sal_fresh  --start 0 --n 600 --mode pair
rd select_eval/sal_both_600   --split sal_both   --start 0 --n 600 --mode pair
rd select_eval/attr_fresh_300 --split attr_fresh --start 0 --n 300 --mode pair
echo "FRESH_RENDER_DONE $(date)"
for m in SELECT_SEEDOOD_DONE SELECT_MIX_DONE SELECT_FAR_DONE SELECT_CTLSEEDS_DONE SELECT_COND_DONE SELECT_SCALE_DONE; do
  until grep -q "$m" results/lane_select_*.log 2>/dev/null; do sleep 120; done
done
CK=$DATA/select_ckpt
ARMS_A="r0_rho00_film r0_rho90_film r1_film r1u_film r0_rho00_film_s1 r0_rho90_film_s1 r1_film_s1 r1u_film_s1 r0_rho00_film_s2 r0_rho90_film_s2 r1_film_s2 r1u_film_s2 r0_rho00_film_s3 r0_rho90_film_s3 r1_film_s3 r1u_film_s3 r0_rho00_film_s4 r0_rho90_film_s4 r1_film_s4 r1u_film_s4"
ARMS_B="r0_rho50_film r0_rho97_film r0_rho99_film r0_rho50_film_s1 r0_rho97_film_s1 r0_rho99_film_s1 r0_rho50_film_s2 r0_rho97_film_s2 r0_rho99_film_s2"
ARMS_C="r1n_film r0p_film r1n_film_s1 r0p_film_s1 r0_rho00_far146 mix_rho90_paired mix_rho90_natural r1_late r1_token r0_rho00_late r0_rho90_late r0_rho00_film_n4000 r1_film_n4000 r1_blank_film r0_rho00_far146_s1 r0_rho00_far146_s2 r1_late_s1 r1_late_s2 r1_token_s1 r1_token_s2 mix_rho90_paired_s1 mix_rho90_paired_s2 mix_rho90_natural_s1 mix_rho90_natural_s2 mix_rho90_biased"
SETS="iidf:$DATA/select_eval/iid_fresh_600 salf:$DATA/select_eval/sal_fresh_600 salb:$DATA/select_eval/sal_both_600 attrf:$DATA/select_eval/attr_fresh_300"
$PYT scripts/eval_select_batch.py --ckpts $ARMS_A $ARMS_B $ARMS_C --sets $SETS 2>&1 | grep -v "^F:\|warnings.warn"
echo "SELECT_FRESH_DONE $(date)"
