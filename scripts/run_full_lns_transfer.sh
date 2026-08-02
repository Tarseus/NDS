#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?run directory}
GPU_ID=${2:-3}

ROOT=/data1/gushengda/NDS_crossdist
PYTHON_BIN=$ROOT/.venv/bin/python

mkdir -p "$RUN_DIR"
{
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "host=$(hostname)"
  echo "gpu=$GPU_ID"
  echo "started_at=$(date -Is)"
  echo "command=run_full_lns_transfer.py --instances 24 --iterations 200 --rollout-size 32"
} > "$RUN_DIR/launcher.log"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" run_full_lns_transfer.py \
  --output "$RUN_DIR" \
  --instances 24 \
  --iterations 200 \
  --rollout-size 32 \
  --seed 20260806 \
  --random-code-seed 20260807 \
  --code-seed 20260803 \
  --num-processes 8 \
  2>&1 | tee "$RUN_DIR/log.txt"

"$PYTHON_BIN" analyze_full_lns_transfer.py "$RUN_DIR" \
  --bootstrap-samples 10000 \
  --seed 20260808 \
  2>&1 | tee "$RUN_DIR/analysis.log"

echo "finished_at=$(date -Is)" >> "$RUN_DIR/launcher.log"
