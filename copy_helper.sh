#!/usr/bin/env bash
set -euo pipefail

# Run this script from the home-filesystem checkout. It copies the checkout to
# node-local storage and runs every command from that copy.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${AUTH_TRAINING_WORK_ROOT:-/local/bys2107/research}"
DST="$WORK_ROOT/auth-training"
PROFILE="${1:-smoke}"

case "$PROFILE" in
  smoke|baseline|initial|capability_only|attack_heavy|diverse_attack|authorization_balanced|generalization_eval|sync) ;;
  *) echo "Usage: $0 {smoke|baseline|initial|capability_only|attack_heavy|diverse_attack|authorization_balanced|generalization_eval|sync}" >&2; exit 2 ;;
esac

ARTIFACT_ROOT="$DST/artifacts"
HF_CACHE_DIR="$ARTIFACT_ROOT/huggingface"
OUT_ROOT="$ARTIFACT_ROOT/runs"
WANDB_DIR="$ARTIFACT_ROOT/wandb"
TORCH_HOME="$ARTIFACT_ROOT/torch"
XDG_CACHE_HOME="$ARTIFACT_ROOT/cache"
TMP_DIR="$ARTIFACT_ROOT/tmp"
REPORT_ROOT="$SRC/outputs"

sync_reports() {
  mkdir -p "$REPORT_ROOT"
  if [[ -d "$OUT_ROOT" ]]; then
    # Regenerate lightweight plots when the source checkout has plotting code.
    while IFS= read -r -d '' run_dir; do
      PYTHONPATH="$SRC" python "$SRC/plot_results.py" --run-dir "$run_dir" || true
    done < <(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)
    # Copy only human-readable reports; never copy checkpoints, model weights,
    # optimizer state, or Hugging Face cache back to the home filesystem.
    rsync -a --prune-empty-dirs \
      --include '*/' --include '*.png' --include '*.json' --include '*.jsonl' \
      --exclude '*' "$OUT_ROOT/" "$REPORT_ROOT/"
    # Remove only the obsolete report location used before generalization
    # splits were appended to the canonical final evaluation suite.
    while IFS= read -r -d '' legacy; do
      case "$legacy" in
        "$REPORT_ROOT"/*/eval_generalization|"$REPORT_ROOT"/*/plots/eval_generalization) ;;
        *) echo "Refusing to remove unexpected legacy report path: $legacy" >&2; exit 2 ;;
      esac
      rm -rf "$legacy"
    done < <(find "$REPORT_ROOT" -type d \( -name eval_generalization -o -path '*/plots/eval_generalization' \) -print0)
    PYTHONPATH="$SRC" python "$SRC/plot_comparison.py" \
      --runs-root "$OUT_ROOT" --output "$REPORT_ROOT/model_comparison.png" || true
  fi
  echo "Home-visible reports: $REPORT_ROOT"
}

if [[ "$PROFILE" == "sync" ]]; then
  sync_reports
  exit 0
fi

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
  --exclude 'outputs/' \
  --exclude '*.pyc' --exclude '.pytest_cache/' --exclude '.venv/' \
  --exclude 'venv/' "$SRC/" "$DST/"

cd "$DST"
export PYTHONPATH="$DST:${PYTHONPATH:-}"
export HF_HOME="$HF_CACHE_DIR"
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

case "$PROFILE" in
  smoke) bash scripts/smoke_test.sh ;;
  baseline) bash scripts/eval_baseline.sh ;;
  initial)
    bash scripts/eval_baseline.sh
    bash scripts/run_full_matrix.sh
    ;;
  capability_only|attack_heavy|diverse_attack|authorization_balanced) bash scripts/train_full.sh "$PROFILE" ;;
  generalization_eval) bash scripts/eval_generalization_matrix.sh ;;
esac

sync_reports

echo "Completed profile: $PROFILE"
echo "Artifacts: $ARTIFACT_ROOT"
