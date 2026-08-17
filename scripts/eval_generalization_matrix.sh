#!/usr/bin/env bash
set -euo pipefail

# Add the two deconfounded splits to each run's single canonical evaluation set.
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
  if [[ "${run_name}" == baseline__* ]]; then
    out="${run}"
  else
    out="${run}/eval_final"
  fi
  python evaluate.py \
    --data-dir "${DATA_DIR}" --model "${model_ref}" --output-dir "${out}" \
    --hf-cache-dir "${HF_CACHE_DIR}" --splits "${SPLITS[@]}" --batch-size 8 --max-new-tokens 128 --append

  # Earlier revisions placed these same results in a second evaluation
  # directory. They are now merged into the canonical evaluation suite.
  legacy_eval="${run}/eval_generalization"
  legacy_plot="${run}/plots/eval_generalization"
  for legacy in "${legacy_eval}" "${legacy_plot}"; do
    case "${legacy}" in
      "${run}"/eval_generalization|"${run}"/plots/eval_generalization) ;;
      *) echo "Refusing to remove unexpected legacy path: ${legacy}" >&2; exit 2 ;;
    esac
    [[ -e "${legacy}" ]] && rm -rf "${legacy}"
  done
done
