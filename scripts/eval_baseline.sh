#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-authorization_dataset_v0/data/generated}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
OUT="${OUT:-$ARTIFACT_ROOT/runs/baseline__${MODEL##*/}}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ARTIFACT_ROOT/huggingface}"

OUT_ROOT="${OUT_ROOT:-$ARTIFACT_ROOT/runs}"
case "$OUT" in
  "$OUT_ROOT"/*) ;;
  *) echo "Refusing to remove output outside OUT_ROOT: $OUT" >&2; exit 2 ;;
esac
if [[ -e "$OUT" ]]; then
  echo "Removing previous baseline output: $OUT"
  rm -rf "$OUT"
fi
echo "Starting fresh baseline evaluation: $OUT"

python evaluate.py --data-dir "$DATA_DIR" --model "$MODEL" --output-dir "$OUT" \
  --hf-cache-dir "$HF_CACHE_DIR" --splits all --batch-size 8 --max-new-tokens 128
