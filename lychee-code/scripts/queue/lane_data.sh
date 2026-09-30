#!/bin/bash
# data lane: follow-up datasets (bias dial rho = 0.9 / 0.5, independent-command pairs) after the main unpaired set is done
source "$(dirname "$0")/common.sh"
until [ -f $DATA/train_unpaired/shard_manifest.json ]; do sleep 30; done
echo "main unpaired set done $(date)"
cache_of unpaired train_unpaired cache_unpaired
gen train_b90 0 1200 0 train_b90 5;   cache_of b90 train_b90 cache_train_b90
gen train 0 600 2 train_indep 5;      cache_of indep train_indep cache_train_indep
gen train_b50 0 1200 0 train_b50 5;   cache_of b50 train_b50 cache_train_b50
echo "DATA_LANE_DONE $(date)"
