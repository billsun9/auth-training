# Authorization SFT + Eval Pipeline

This package is designed for the **single canonical dataset layout**:

```text
authorization_dataset_v0/data/generated/
  train_attack_heavy.jsonl
  train_diverse_attack.jsonl
  train_authorization_balanced.jsonl
  eval_iid.jsonl
  eval_lexical_ood.jsonl
  eval_mechanism_ood.jsonl
  eval_auth_recombination.jsonl
  eval_benign_control.jsonl
```

There is deliberately no support for the older nested-per-regime eval layout.

## Objective

Every row already has a fully rendered `prompt` and a JSON-object `target`. SFT trains:

```text
<prompt>\n<target JSON><eos>
```

with prompt labels masked to `-100`, i.e. **completion-only cross-entropy loss**. The model is never given `authorized`, `style`, `mechanism`, `is_attack`, or `attack_like` as extra labels; those fields are used only for analysis.

No additional Qwen chat template is wrapped around the generated prompt. This preserves the exact training/eval prompt distribution.

## Install

```bash
pip install -r requirements.txt
```

## Validate data

```bash
python validate_data.py
```

Checks the required schema, canonical filenames, split/regime labels, uniqueness within files, and both ID-level and exact prompt+target train/eval disjointness. Training regimes are allowed to overlap with each other.

## Smoke test

Tiny end-to-end **train + all-five-splits eval**:

```bash
bash scripts/smoke_test.sh
```

Defaults to Qwen2.5-0.5B-Instruct + LoRA, 16 train examples, 2 optimizer steps, and 8 examples per eval split. This only verifies plumbing; its metrics are not meaningful.

## Main experiment: Qwen2.5-0.5B-Instruct

One regime:

```bash
bash scripts/train_full.sh authorization_balanced
```

The first comparison uses the attack-heavy and authorization-balanced regimes
with identical settings:

```bash
bash scripts/run_full_matrix.sh
```

Default main settings:
- full-parameter SFT
- BF16
- 2 GPUs via `torchrun` (`NPROC_PER_NODE=1` is supported for a single GPU)
- global batch = 2 examples/GPU × 2 GPUs × 4 grad accumulation = 16
- 2 epochs
- LR 2e-5
- gradient checkpointing
- checkpoints every 20 optimizer steps
- greedy eval on the same five shared files

## Frozen baseline

```bash
bash scripts/eval_baseline.sh
```

## Qwen2.5-3B full SFT on 2×48GB

Use full-shard FSDP:

```bash
bash scripts/train_full_3b_fsdp.sh authorization_balanced
```

Use the same optimization method/settings for every regime being compared. For a cheap 7B pilot, `METHOD=lora` is supported, but don't compare a LoRA-trained regime against a full-FT regime as if training method were controlled.

## Logs

Each run writes:
- `run_config.json`
- `logs/train_log.jsonl`
- `logs/trainer_log_history.jsonl`
- `train_metrics.json`
- `checkpoint-*` (model weights plus optimizer, scheduler, RNG, and trainer state)
- `final/`
- `eval_final/` (predictions and metrics when using the provided scripts)

Downloaded Hugging Face model/tokenizer files go to `artifacts/huggingface` by
default. Training checkpoints, final model weights, and evaluation outputs go
under `artifacts/runs`. Set one root for the whole experiment, for example
`ARTIFACT_ROOT=D:/auth-artifacts`; use `HF_CACHE_DIR` and `OUT_ROOT` for
separate overrides. The Python entry points expose the cache setting as
`--hf-cache-dir` and the output setting as `--output-dir`.
When W&B is enabled, its local run files go to `artifacts/wandb` (or
`WANDB_DIR`/`--wandb-dir`); W&B may also maintain its own small user cache.
The Python entry points expose the same setting as `--hf-cache-dir`.

Optional W&B:

```bash
WANDB=1 bash scripts/train_full.sh authorization_balanced
```

## Evaluation outputs

Each split writes `predictions_<split>.jsonl` and `metrics_<split>.json`; all metrics are combined in `eval_summary.json`.

Core metrics:
- JSON parse rate
- exact **structured target** accuracy (JSON object equality; formatting/key-order don't matter)
- action accuracy
- authorized exact accuracy
- unauthorized exact accuracy
- **unauthorized_execution_rate**: unauthorized examples where the model executes `candidate_action`
- counterfactual-pair exact/action accuracy
- group accuracy by source/style/mechanism/candidate action/domain

## Learning dynamics

```bash
python evaluate_checkpoints.py \
  --run-dir runs/authorization_balanced__Qwen2.5-0.5B-Instruct__full__seed0 \
  --data-dir authorization_dataset_v0/data/generated
```

This evaluates every `checkpoint-*` plus `final` against the same shared eval suite.
