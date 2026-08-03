#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/gushengda/NDS_dynamic_transport
RUN_ID=${1:?run id is required}
RUN_DIR=/data1/gushengda/NDS_dynamic_transport_runs/${RUN_ID}
PYTHON_BIN=/data1/gushengda/NDS_crossdist/.venv/bin/python
SOURCE_MEMORY=/data1/gushengda/NDS_crossdist_runs/multisource-confirm-20260803-073235/raw_metrics.npz

mkdir -p "$RUN_DIR"
{
  echo "run_id=$RUN_ID"
  echo "source_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "checkpoint=models/cvrp_100/checkpoint-2000.pt"
  echo "source_memory=$SOURCE_MEMORY"
  echo "target_code_mode=resampled"
  echo "fixed_target_codes=false"
} > "$RUN_DIR/launcher.log"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" run_dynamic_code_transport.py \
  --source-memory "$SOURCE_MEMORY" \
  --output "$RUN_DIR" \
  --instances 16 \
  --rounds 4 \
  --rollout-size 32 \
  --num-processes 8 \
  2>&1 | tee "$RUN_DIR/log.txt"

"$PYTHON_BIN" analyze_dynamic_code_transport.py "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/log.txt"
