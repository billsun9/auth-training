#!/usr/bin/env bash
set -euo pipefail

# Evaluate already-trained final checkpoints on only the two deconfounded splits.
DATA_DIR="${DATA_DIR:-authorization_dataset_v0/data/generated}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
OUT_ROOT="${OUT_ROOT:-$ARTIFACT_ROOT/runs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ARTIFACT_ROOT/huggingface}"
MODEL_SLUG="${MODEL##*/}"
SPLITS=(auth_recombination_natural authorization_policy_ood)

declare -a RUNS=(
  "baseline__${MODEL_SLUG}"
  "attack_heavy__${MODEL_SLUG}__full__seed0"
  "diverse_attack__${MODEL_SLUG}__full__seed0"
  "authorization_balanced__${MODEL_SLUG}__full__seed0"
  "capability_only__${MODEL_SLUG}__full__seed0"
)

for run_name in "${RUNS[@]}"; do
  run="${OUT_ROOT}/${run_name}"
  if [[ ! -d "${run}" ]]; then
    echo "Skipping absent run: ${run_name}"
    continue
  fi
  if [[ "${run_name}" == baseline__* ]]; then
    model_ref="${MODEL}"
  else
    [[ -d "${run}/final" ]] || { echo "Skipping run without final/: ${run_name}" >&2; continue; }
    model_ref="${run}/final"
  fi
  out="${run}/eval_generalization"
  case "${out}" in
    "${run}"/eval_generalization) ;;
    *) echo "Refusing to remove unexpected evaluation output: ${out}" >&2; exit 2 ;;
  esac
  rm -rf "${out}"
  python evaluate.py \
    --data-dir "${DATA_DIR}" --model "${model_ref}" --output-dir "${out}" \
    --hf-cache-dir "${HF_CACHE_DIR}" --splits "${SPLITS[@]}" --batch-size 8 --max-new-tokens 128
done
