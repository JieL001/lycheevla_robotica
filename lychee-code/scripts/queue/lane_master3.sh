#!/bin/bash
# Runs next to the tail of lane_master2.sh: as soon as the far and ctlseeds lanes have released the GPU (master2 then works on the simulator-bound closed-loop lane and the rendering), train seeds 3 and 4 of the four arms of
# the planned comparisons and seeds 1 and 2 of the other dial settings, and finally evaluate the new checkpoints on the fresh scenes (the fresh lane skips finished files).
source "$(dirname "$0")/common.sh"
until grep -q "SELECT_CLOSED2_DONE" results/lane_select_closed2.log 2>/dev/null; do sleep 60; done
step() { echo "=== step $1 $(date)"; bash scripts/queue/lane_select_$1.sh >> results/lane_select_$1.log 2>> results/lane_select_$1.err; }
step seeds34
step dialseeds
until grep -q "MASTER2_DONE" results/lane_master2.log 2>/dev/null; do sleep 60; done
step fresh
echo "MASTER3_DONE $(date)"
