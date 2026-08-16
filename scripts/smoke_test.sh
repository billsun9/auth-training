#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_DIR="${DATA_DIR:-authorization_dataset_v0/data/generated}"
MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
OUT="${SMOKE_OUT:-$ARTIFACT_ROOT/runs/smoke_authorization_balanced}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ARTIFACT_ROOT/huggingface}"
WANDB_DIR="${WANDB_DIR:-$ARTIFACT_ROOT/wandb}"
python validate_data.py --data-dir "$DATA_DIR"
python train_sft.py \
  --data-dir "$DATA_DIR" --regime authorization_balanced --model "$MODEL" \
  --method lora --output-dir "$OUT" --hf-cache-dir "$HF_CACHE_DIR" --wandb-dir "$WANDB_DIR" --max-train-samples 16 --max-steps 2 \
  --per-device-train-batch-size 2 --gradient-accumulation-steps 1 \
  --max-seq-length 1024 --logging-steps 1 --save-steps 0 --seed 0
python evaluate.py \
  --data-dir "$DATA_DIR" --model "$OUT/final" --output-dir "$OUT/eval_smoke" --hf-cache-dir "$HF_CACHE_DIR" \
  --splits all --max-samples-per-split 8 --batch-size 4 --max-new-tokens 96 --seed 0
echo "Smoke test passed. Metrics: $OUT/eval_smoke/eval_summary.json"
