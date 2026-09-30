#!/bin/bash
# items for the web-based human validation (one simulator process)
source "$(dirname "$0")/common.sh"
[ -f results/human_check_web/items.json ] && exit 0
acquire 4
$PY -u scripts/make_human_check_web.py --n 200 --out results/human_check_web 2>&1 | grep -v "^F:\|warnings.warn"
$PY scripts/build_human_check_web.py
echo "HUMAN_LANE_DONE $(date)"
