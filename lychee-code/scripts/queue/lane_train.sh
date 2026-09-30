#!/bin/bash
# training lane: one arm at a time, in priority order; arms whose dataset is not ready yet wait for it
source "$(dirname "$0")/common.sh"
waitfor() { until [ -f $DATA/$1/episodes.json ]; do sleep 30; done; }
train r5_full cache_paired --film 0 --ground 1 --inject 1 --ground_sup 1 --lang_direct 0
waitfor cache_unpaired
train r0_film cache_unpaired --film 1 --ground 0
train r0_late cache_unpaired --film 0 --ground 0
train r1_late cache_paired --film 0 --ground 0
train r5_sup_only cache_paired --film 0 --ground 1 --inject 0 --ground_sup 1 --lang_direct 1
train r5_inj_only cache_paired --film 0 --ground 1 --inject 1 --ground_sup 0 --lang_direct 0
train r5_full_lang cache_paired --film 0 --ground 1 --inject 1 --ground_sup 1 --lang_direct 1
train r1_film_blank cache_paired --film 1 --ground 0 --blank 1
waitfor cache_train_b90
train r0_b90_film cache_train_b90 --film 1 --ground 0
waitfor cache_train_indep
train r1u_film cache_train_indep --film 1 --ground 0
waitfor cache_train_b50
train r0_b50_film cache_train_b50 --film 1 --ground 0
train r0_film_n300 cache_unpaired --film 1 --ground 0 --max_eps 300
train r0_film_n600 cache_unpaired --film 1 --ground 0 --max_eps 600
echo "TRAIN_LANE_DONE $(date)"
