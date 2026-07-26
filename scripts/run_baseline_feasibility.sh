#!/usr/bin/env bash
set -euo pipefail

TASK_NAME=${1:?task name}
RUN_DIR=${2:?run directory}
CONFIG_NAME=${3:?training config name}
GPU_ID=${4:?visible physical GPU id}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${NDS_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_BIN=${NDS_PYTHON:-$ROOT/.venv/bin/python}

mkdir -p "$RUN_DIR"
ln -sfn "$ROOT/configs" "$RUN_DIR/configs"
ln -sfn "$ROOT/data" "$RUN_DIR/data"
ln -sfn "$ROOT/src" "$RUN_DIR/src"
ln -sfn "$ROOT/train_baseline_feasibility.py" \
  "$RUN_DIR/train_baseline_feasibility.py"

echo "task=$TASK_NAME"
echo "run_dir=$RUN_DIR"
echo "config=$CONFIG_NAME"
echo "gpu=$GPU_ID"
echo "repo=$ROOT"
echo "commit=$(git -C "$ROOT" rev-parse HEAD)"
echo "started_at=$(date -Is)"

cd "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="$RUN_DIR/wandb"

exec "$PYTHON_BIN" train_baseline_feasibility.py "$CONFIG_NAME" \
  "trainer_params.dpp_objective.enabled=false" \
  "+trainer_params.feasibility_diagnostics.enabled=true" \
  "+trainer_params.feasibility_diagnostics.probe_enabled=true" \
  "+trainer_params.feasibility_diagnostics.subset_sizes=[8,16,32,64]" \
  "+trainer_params.feasibility_diagnostics.panel_size=64" \
  "+trainer_params.feasibility_diagnostics.probe_interval=97" \
  "+trainer_params.feasibility_diagnostics.probe_seed=20260726" \
  "+logger_params.filepath=$RUN_DIR" \
  "+logger_params.filename=log.txt" \
  "trainer_params.model_save_interval=10"
