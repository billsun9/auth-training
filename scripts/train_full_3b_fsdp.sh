#!/usr/bin/env bash
set -euo pipefail

# Full-parameter Qwen2.5-3B SFT on 2x48GB using Accelerate FSDP2.
# FSDP activation checkpointing is enabled in the config, so ordinary
# Transformers gradient checkpointing is disabled here to avoid double-checkpointing.

REGIME="${1:?Usage: bash scripts/train_full_3b_fsdp.sh <regime>}"
DATA_DIR="${DATA_DIR:-authorization_dataset_v0/data/generated}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-2}"
OUT_ROOT="${OUT_ROOT:-runs}"
MODEL_SLUG="${MODEL##*/}"
OUT="$OUT_ROOT/${REGIME}__${MODEL_SLUG}__full__seed${SEED}"

python validate_data.py --data-dir "$DATA_DIR"

accelerate launch --config_file configs/accelerate_fsdp2_2gpu.yaml train_sft.py \
  --data-dir "$DATA_DIR" \
  --regime "$REGIME" \
  --model "$MODEL" \
  --method full \
  --output-dir "$OUT" \
  --num-train-epochs "$EPOCHS" \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 1024 \
  --no-gradient-checkpointing \
  --logging-steps 5 \
  --save-steps 20 \
  --save-total-limit 1 --save-only-model \
  --seed "$SEED"

python evaluate.py \
  --data-dir "$DATA_DIR" \
  --model "$OUT/final" \
  --output-dir "$OUT/eval_final" \
  --splits all \
  --batch-size 8 \
  --max-new-tokens 128 \
  --seed "$SEED"

echo "Finished: $OUT"
