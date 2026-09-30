#!/bin/bash
# one long end-to-end run: does the compact behaviour-cloning policy learn to harvest at all when trained 7.5x longer?
# (the 8-epoch R1/FiLM policy harvested 0 of 80 episodes; see README)
source "$(dirname "$0")/common.sh"
EPOCHS=60 train r1_film_long cache_paired --film 1 --ground 0
if saved r1_film_long; then
  acquire 9
  w=$(pick_workers)
  echo "== eval r1_film_long iid n=40 workers=$w $(date)"
  $PY -u scripts/eval_policy.py --policy "bc|$DATA/ckpt/r1_film_long.pt|4" --obs rgb --split iid --n 40 --workers $w --out results/eval/bc_r1_film_long__iid.jsonl 2>&1 | grep -v "^F:\|warnings.warn" | grep -v "^    "
fi
echo "E2E_LANE_DONE $(date)"
