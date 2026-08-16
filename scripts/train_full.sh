#!/usr/bin/env bash
set -euo pipefail
REGIME="${1:?Usage: bash scripts/train_full.sh <attack_heavy|diverse_attack|authorization_balanced>}"
DATA_DIR="${DATA_DIR:-authorization_dataset_v0/data/generated}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
METHOD="${METHOD:-full}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
OUT_ROOT="${OUT_ROOT:-$ARTIFACT_ROOT/runs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ARTIFACT_ROOT/huggingface}"
WANDB_DIR="${WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
MODEL_SLUG="${MODEL##*/}"
OUT="$OUT_ROOT/${REGIME}__${MODEL_SLUG}__${METHOD}__seed${SEED}"
EXTRA=(--wandb-dir "$WANDB_DIR")
[[ "${WANDB:-0}" == "1" ]] && EXTRA+=(--wandb)

# A rerun is deliberately fresh: never resume from an old checkpoint or mix
# reports from two attempts. The guard limits deletion to this run root.
case "$OUT" in
  "$OUT_ROOT"/*) ;;
  *) echo "Refusing to remove output outside OUT_ROOT: $OUT" >&2; exit 2 ;;
esac
if [[ -e "$OUT" ]]; then
  echo "Removing previous run output: $OUT"
  rm -rf "$OUT"
fi
echo "Starting fresh full-SFT run: $OUT"

python validate_data.py --data-dir "$DATA_DIR"
TRAIN_CMD=(python train_sft.py)
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  TRAIN_CMD=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_sft.py)
fi
"${TRAIN_CMD[@]}" \
  --data-dir "$DATA_DIR" --regime "$REGIME" --model "$MODEL" --method "$METHOD" \
  --output-dir "$OUT" --hf-cache-dir "$HF_CACHE_DIR" --num-train-epochs "$EPOCHS" \
  --per-device-train-batch-size 2 --gradient-accumulation-steps 4 \
  --max-seq-length 1024 --validation-ratio 0.1 --eval-steps 20 --early-stopping-patience 5 \
  --logging-steps 5 --save-steps 20 --save-total-limit 3 \
  --seed "$SEED" "${EXTRA[@]}"
python evaluate.py \
  --data-dir "$DATA_DIR" --model "$OUT/final" --output-dir "$OUT/eval_final" --hf-cache-dir "$HF_CACHE_DIR" \
  --splits all --batch-size 8 --max-new-tokens 128 --seed "$SEED"
echo "Finished: $OUT"
