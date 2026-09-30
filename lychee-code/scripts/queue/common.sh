#!/bin/bash
# Shared helpers for the background pipeline (Windows laptop: 15 GB RAM, 61 GB commit limit -- everything is gated on FREE COMMIT,
# because a process that cannot be committed dies with WinError 1455 / "Couldn't open shared file mapping").
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1     # the lychee-code/ directory (this file is lychee-code/scripts/queue/common.sh)
export PYTHONIOENCODING=utf-8
PY=/f/miniforge3/envs/lychee/python.exe        # simulator, data generation, evaluation
PYT=/f/miniforge3/envs/relcomp/python.exe      # CUDA torch for training (read-only use of that env)
DATA=D:/lychee_data

memfree() { powershell.exe -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -d '\r'; }
wait_mem() { while [ "$(memfree)" -lt "$1" ]; do sleep 30; done; }
# acquire <GB>: serialise launches, so that several lanes cannot pass the memory gate in the same minute; the lock is released
# 50 s after the gate opens (the time a new process needs to reach its full commit)
acquire() {
  local lock=results/.launchlock
  until mkdir $lock 2>/dev/null; do
    [ -n "$(find $lock -maxdepth 0 -mmin +5 2>/dev/null)" ] && rmdir $lock 2>/dev/null     # stale lock
    sleep 7
  done
  wait_mem $1
  ( sleep 50; rmdir $lock 2>/dev/null ) &
}
saved() { grep -q "^saved" results/train_$1.log 2>/dev/null; }

train() {  # train <name> <cache dir under $DATA> <train_bc.py flags...>   (skips finished arms, retries on failure)
  local name=$1 cache=$2; shift 2
  saved $name && return 0
  for attempt in 1 2 3 4 5 6; do
    acquire 8
    echo "== train $name (attempt $attempt) $(date)"
    $PYT -u scripts/train_bc.py --cache $DATA/$cache --out $DATA/ckpt/$name.pt --epochs ${EPOCHS:-8} --bs 128 --workers 2 "$@" > results/train_$name.log 2>&1
    if saved $name; then tail -n 2 results/train_$name.log; return 0; fi
    echo "   attempt $attempt failed: $(tail -n 1 results/train_$name.log | cut -c1-160)"; sleep 90
  done
  return 1
}

cache_of() {  # cache_of <name> <shard dir under $DATA> <cache dir under $DATA>
  [ -f $DATA/$3/episodes.json ] && return 0
  acquire 6
  $PY -u scripts/build_cache.py --shards "$DATA/$2/shard_*.h5" --out $DATA/$3 2>&1 | grep -v "^F:\|warnings.warn" | tail -2
}

gen() {  # gen <split> <start> <n> <pair 0|1|2> <dir under $DATA> <workers>
  [ -f $DATA/$5/shard_manifest.json ] && return 0
  acquire 12
  echo "== generate $1 [$2,+$3) pair=$4 -> $5 $(date)"
  $PY -u scripts/generate_data.py --split $1 --start $2 --n $3 --pair $4 --workers $6 --out $DATA/$5/shard 2>&1 | grep --line-buffered -v "^F:\|warnings.warn"
}

pick_workers() {  # simulator workers for an evaluation, from the free commit at launch: ~2 GB per worker + 1.5 GB main
  local f; f=$(memfree)
  if [ "$f" -ge 20 ]; then echo 6; elif [ "$f" -ge 14 ]; then echo 4; else echo 3; fi
}
