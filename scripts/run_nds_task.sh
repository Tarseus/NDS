#!/usr/bin/env bash
set -euo pipefail

TASK_NAME=${1:?task name}
RUN_DIR=${2:?run directory}
CONFIG_NAME=${3:?training config name}
GPU_ID=${4:?visible physical GPU id}
WAIT_FOR=${5:-}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${NDS_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${NDS_PYTHON:-$ROOT/.venv/bin/python}

mkdir -p "$RUN_DIR"
ln -sfn "$ROOT/configs" "$RUN_DIR/configs"
ln -sfn "$ROOT/data" "$RUN_DIR/data"
ln -sfn "$ROOT/src" "$RUN_DIR/src"
ln -sfn "$ROOT/train.py" "$RUN_DIR/train.py"
ln -sfn "$ROOT/eval.py" "$RUN_DIR/eval.py"

echo "task=$TASK_NAME"
echo "run_dir=$RUN_DIR"
echo "config=$CONFIG_NAME"
echo "gpu=$GPU_ID"
echo "repo=$ROOT"
echo "commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "started_at=$(date -Is)"

if [ -n "$WAIT_FOR" ]; then
  echo "waiting_for=$WAIT_FOR/checkpoint-1500.pt"
  while [ ! -f "$WAIT_FOR/checkpoint-1500.pt" ]; do
    echo "wait_status=$(date -Is) checkpoint_missing=$WAIT_FOR/checkpoint-1500.pt"
    sleep 300
  done
  echo "wait_status=$(date -Is) predecessor_checkpoint_ready"
fi

cd "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="$RUN_DIR/wandb"

OVERRIDES=(
  "+logger_params.filepath=$RUN_DIR"
  "+logger_params.filename=log.txt"
  "trainer_params.model_save_interval=10"
)
if [ -n "$WAIT_FOR" ]; then
  OVERRIDES+=("trainer_params.model_load.path=$WAIT_FOR")
fi

exec "$PYTHON_BIN" train.py "$CONFIG_NAME" "${OVERRIDES[@]}"
