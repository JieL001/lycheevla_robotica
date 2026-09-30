#!/bin/bash
# Sensitivity of the selection-track results to the snap radius of the pixel-to-fruit rule (20 px in all reported results): the arms of the planned comparisons, the two
# strongest dial settings and the late-fusion / command-token variants, evaluated with radii of 10, 30 and 40 px on the IID, saliency-reversed and attribute-OOD scenes
# (files s_<arm>_snap<R>__<split>.jsonl).  Evaluation only (about 0.5 GB of GPU memory), so it can run next to a training job; it waits for enough free memory.
source "$(dirname "$0")/common.sh"
CK=$DATA/select_ckpt
EV=$DATA/select_eval
SETS="iid:$EV/iid_600 sal:$EV/sal_600 attr:$EV/attr_300"
ARMS="r0_rho00_film r0_rho00_film_s1 r0_rho00_film_s2 r0_rho00_film_s3 r0_rho00_film_s4 r0_rho90_film r0_rho90_film_s1 r0_rho90_film_s2 r0_rho90_film_s3 r0_rho90_film_s4
 r1_film r1_film_s1 r1_film_s2 r1_film_s3 r1_film_s4 r1u_film r1u_film_s1 r1u_film_s2 r1u_film_s3 r1u_film_s4
 r0_rho97_film r0_rho97_film_s1 r0_rho97_film_s2 r0_rho99_film r0_rho99_film_s1 r0_rho99_film_s2 r1_late r1_late_s1 r1_late_s2 r1_token r1_token_s1 r1_token_s2"
for R in 10 30 40; do
  acquire 4
  $PYT scripts/eval_select_batch.py --ckpt_dir $CK --ckpts $ARMS --sets $SETS --snap $R --suffix _snap$R 2>&1 | grep -v "^F:\|warnings.warn" | cut -c1-170
done
echo "SNAP_DONE $(date)"
