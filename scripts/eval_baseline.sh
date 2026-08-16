#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-authorization_dataset_v0/data/generated}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
OUT="${OUT:-$ARTIFACT_ROOT/runs/baseline__${MODEL##*/}}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ARTIFACT_ROOT/huggingface}"
python evaluate.py --data-dir "$DATA_DIR" --model "$MODEL" --output-dir "$OUT" \
  --hf-cache-dir "$HF_CACHE_DIR" --splits all --batch-size 8 --max-new-tokens 128
