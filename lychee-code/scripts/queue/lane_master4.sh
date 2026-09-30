#!/bin/bash
# Last part of the queue (after lane_master3.sh): more seeds for the single-seed arms, the biased-plus-biased control, the end-to-end language probes, and a final evaluation of every
# new checkpoint on the fresh scenes.
source "$(dirname "$0")/common.sh"
until grep -q "MASTER3_DONE" results/lane_master3.log 2>/dev/null; do sleep 60; done
step() { echo "=== step $1 $(date)"; bash scripts/queue/lane_$1.sh >> results/lane_$1.log 2>> results/lane_$1.err; }
step select_ctl2seeds
step select_mixbiased
step e2e_probes
step select_fresh
echo "MASTER4_DONE $(date)"
