#!/bin/bash
# measurements quoted in the manuscript (single simulator process each, gated on free commit)
source "$(dirname "$0")/common.sh"
acquire 6; $PY -u scripts/check_visibility.py 60 off 2>&1 | grep -v "^F:\|warnings.warn" | tee results/visibility_check_off.log
acquire 5; $PY -u scripts/sim_constants.py 2>&1 | grep -v "^F:\|warnings.warn" | tee results/sim_constants.log
acquire 3; $PY -u scripts/bias_dial_stats.py 300 2>&1 | grep -v "^F:\|warnings.warn" | tee results/bias_dial_stats.log
echo "MISC_LANE_DONE $(date)"
