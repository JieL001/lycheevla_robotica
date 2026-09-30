#!/bin/bash
# Serial tail of the second wave.  Five concurrent jobs made the laptop thrash (free physical memory < 1 GB, no training epoch finished in an hour), so the lanes that had
# not started were stopped and their work runs here, one job at a time, next to the two trainer lanes (scale, mix) that keep going.
source "$(dirname "$0")/common.sh"
step() {  # step <lane name>: run lane_select_<name>.sh with its usual log files
  echo "=== step $1 $(date)"
  bash scripts/queue/lane_select_$1.sh >> results/lane_select_$1.log 2>> results/lane_select_$1.err
}
step seedood
step narms
step far
step ctlseeds
step closed2
step fresh
echo "MASTER2_DONE $(date)"
