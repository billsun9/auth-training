#!/usr/bin/env bash
set -euo pipefail

# Run this script from the home-filesystem checkout. It copies the checkout to
# node-local storage and runs every command from that copy.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${AUTH_TRAINING_WORK_ROOT:-/local/bys2107/research}"
DST="$WORK_ROOT/auth-training"
PROFILE="${1:-smoke}"

case "$PROFILE" in
  smoke|baseline|initial|attack_heavy|authorization_balanced) ;;
  *) echo "Usage: $0 {smoke|baseline|initial|attack_heavy|authorization_balanced}" >&2; exit 2 ;;
esac

ARTIFACT_ROOT="$DST/artifacts"
HF_CACHE_DIR="$ARTIFACT_ROOT/huggingface"
OUT_ROOT="$ARTIFACT_ROOT/runs"
WANDB_DIR="$ARTIFACT_ROOT/wandb"
TORCH_HOME="$ARTIFACT_ROOT/torch"
XDG_CACHE_HOME="$ARTIFACT_ROOT/cache"
TMP_DIR="$ARTIFACT_ROOT/tmp"

mkdir -p "$DST" "$ARTIFACT_ROOT" "$HF_CACHE_DIR" "$OUT_ROOT" \
  "$WANDB_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMP_DIR"

echo "Source: $SRC"
echo "Local work directory: $DST"
echo "Profile: $PROFILE"
echo "Node: $(hostname)"
echo "Copying source and datasets to local storage..."

# Do not copy generated run artifacts or developer caches from the source
# checkout. No --delete is used, so prior local checkpoints are preserved.
rsync -a \
  --exclude '.git/' --exclude 'artifacts/' --exclude 'runs/' \
  --exclude 'checkpoints/' --exclude 'wandb/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude '.pytest_cache/' --exclude '.venv/' \
  --exclude 'venv/' "$SRC/" "$DST/"

cd "$DST"
export PYTHONPATH="$DST:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export TORCH_HOME="$TORCH_HOME"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TMPDIR="$TMP_DIR"
export ARTIFACT_ROOT="$ARTIFACT_ROOT"
export HF_CACHE_DIR="$HF_CACHE_DIR"
export OUT_ROOT="$OUT_ROOT"
export WANDB_DIR="$WANDB_DIR"
export DATA_DIR="$DST/authorization_dataset_v0/data/generated"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

python validate_data.py --data-dir "$DST/authorization_dataset_v0/data/generated"
python -c 'import torch; print(f"Torch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}; CUDA build: {torch.version.cuda}"); raise SystemExit("CUDA is unavailable; install a CUDA-compatible PyTorch build before training") if not torch.cuda.is_available() else None'

case "$PROFILE" in
  smoke) bash scripts/smoke_test.sh ;;
  baseline) bash scripts/eval_baseline.sh ;;
  initial)
    bash scripts/eval_baseline.sh
    bash scripts/run_full_matrix.sh
    ;;
  attack_heavy|authorization_balanced) bash scripts/train_full.sh "$PROFILE" ;;
esac

echo "Completed profile: $PROFILE"
echo "Artifacts: $ARTIFACT_ROOT"
