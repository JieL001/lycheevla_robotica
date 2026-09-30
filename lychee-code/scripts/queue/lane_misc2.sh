#!/bin/bash
# Small simulator jobs for the manuscript, run when the free commit allows: visibility check (analytic vs rendered), simulator constants.
source "$(dirname "$0")/common.sh"
acquire 5
echo "== visibility check $(date)"
$PY -u scripts/check_visibility.py 60 off 2>&1 | grep -v "^F:\|warnings.warn" | tail -n 8
$PY scripts/make_visibility_figure.py 2>&1 | tail -n 2
acquire 4
echo "== simulator constants $(date)"
$PY -u scripts/sim_constants.py 2>&1 | grep -v "^F:\|warnings.warn" | tail -n 25
echo "MISC2_DONE $(date)"
