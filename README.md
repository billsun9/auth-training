# Authorization SFT + Evaluation Pipeline

This repository implements a completion-only causal-LM SFT experiment for the
authorization dataset. Each example already contains a fully rendered prompt
and a JSON-object target. Training serializes:

```text
<prompt><separator><target JSON><eos>
```

Only target and EOS tokens contribute to cross-entropy loss. Prompt and padding
labels are `-100`. Evaluation uses the same prompt serializer, greedy JSON
generation, structural parsing, and shared five-way evaluation suite.

## Repository data

The canonical data directory is:

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

The three training files are separate experimental regimes. Each has 3,000
regime-specific hierarchy rows plus 1,000 byte-identical `shared_capability`
rehearsal rows by default. The five eval files are shared across regimes and
include a held-out closed-domain injected-data subset. `validate_data.py` checks
schema, canonical layout, split/regime labels, duplicate IDs, and exact
prompt+target train/eval leakage.

## Local setup and downloads

Use a Python environment with a CUDA-enabled PyTorch build on the GPU cluster.
Do not replace a working cluster CUDA PyTorch installation with a CPU wheel.
From the repository checkout:

```bash
conda activate nlp                 # or the cluster environment you use
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install -r requirements.txt
python validate_data.py
```

The first smoke or training run downloads its tokenizer and model into the
configured local Hugging Face cache. The smoke test uses Qwen2.5-0.5B-Instruct;
the full baseline and initial comparison default to Qwen2.5-1.5B-Instruct.
On the cluster, the copy helper below puts this cache on `/local`; do not run
the experiment from the home filesystem.

## Slurm / node-local execution

`copy_helper.sh` lives in this repository and is the main entry point for the
cluster workflow. Run it from the home-filesystem checkout, inside an existing
Slurm GPU allocation:

```bash
cd /insomnia001/home/bys2107/research
conda activate nlp
bash auth-training/copy_helper.sh smoke
```

The helper copies source code and datasets to:

```text
/local/bys2107/research/auth-training
```

It then changes into that directory before running anything. All large or
frequently written artifacts stay under:

```text
/local/bys2107/research/auth-training/artifacts/
  huggingface/   # model/tokenizer downloads
  runs/          # checkpoints, final weights, logs, metrics, predictions
  wandb/         # W&B local files, if enabled
  torch/         # Torch cache
  cache/         # general cache
  tmp/           # temporary files
```

The source checkout and its JSONL data are copied to local storage as well, so
training data reads and all experiment writes happen under `/local`. The helper
does not use `rsync --delete`, so unrelated local artifacts and the Hugging
Face cache are preserved. Each baseline or full-SFT script deliberately removes
its own exact output directory before starting, so repeating the same model,
regime, method, and seed is a fresh run rather than a resume.
At the end of each run, it copies only small reports (PNGs, JSON metrics, JSONL
predictions/logs) back to `auth-training/outputs/` in the home checkout. Model
weights, checkpoints, optimizer state, and Hugging Face cache remain local.
Override the local root only when needed:

```bash
AUTH_TRAINING_WORK_ROOT=/local/bys2107/research bash auth-training/copy_helper.sh smoke
```

The helper profiles are:

| Profile | Action |
|---|---|
| `smoke` | 16-example LoRA plumbing test, 2 optimizer steps, all five eval splits with 8 examples each |
| `baseline` | Frozen Qwen2.5-1.5B-Instruct greedy evaluation on all five shared splits |
| `initial` | Baseline, then identical full-SFT runs for `attack_heavy` and `authorization_balanced` |
| `attack_heavy` | One full-SFT run plus evaluation |
| `authorization_balanced` | One full-SFT run plus evaluation |

## Smoke test and results

Run:

```bash
bash auth-training/copy_helper.sh smoke
```

The smoke test is only a plumbing check; its metrics are not scientifically
meaningful. Inspect results from the allocated node:

```bash
SMOKE=/local/bys2107/research/auth-training/artifacts/runs/smoke_authorization_balanced
cat "$SMOKE/eval_smoke/eval_summary.json"
cat "$SMOKE/eval_smoke/metrics_iid.json"
less "$SMOKE/eval_smoke/predictions_iid.jsonl"
```

Check that all five `metrics_*.json` and `predictions_*.jsonl` files exist and
that the run reaches `final/`.

Home-visible copies are under:

```bash
ls -al /insomnia001/home/bys2107/research/auth-training/outputs/smoke_authorization_balanced
```

After manually generating plots or if a run ended before report sync, run:

```bash
bash auth-training/copy_helper.sh sync
```

## Initial experiment and results

The initial meaningful comparison is frozen baseline plus identical full-SFT
runs on `attack_heavy`, `diverse_attack`, and `authorization_balanced`:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
NPROC_PER_NODE=1 \
bash auth-training/copy_helper.sh initial
```

This uses Qwen2.5-1.5B-Instruct, full-parameter SFT, BF16, one GPU by default,
per-device batch size 2, gradient accumulation 4, nominal global batch size
8, two epochs, learning rate `2e-5`, gradient checkpointing, and greedy eval
on the same five shared files. Set `NPROC_PER_NODE=2` when a two-GPU
allocation is available; compared regimes must use the same setting.

Each full-SFT regime holds out a deterministic 10% validation split from its
own training rows (counterfactual pairs stay together). Validation uses the
same completion-only loss, runs every 20 steps, restores the best
`eval_loss` checkpoint, and stops after five validation evaluations without
improvement. The shared five-split suite remains untouched for final reporting.

Results are under:

```text
/local/bys2107/research/auth-training/artifacts/runs/
  baseline__Qwen2.5-1.5B-Instruct/
  attack_heavy__Qwen2.5-1.5B-Instruct__full__seed0/
  diverse_attack__Qwen2.5-1.5B-Instruct__full__seed0/
  authorization_balanced__Qwen2.5-1.5B-Instruct__full__seed0/
```

Inspect the final summaries:

```bash
ROOT=/local/bys2107/research/auth-training/artifacts/runs
cat "$ROOT/baseline__Qwen2.5-1.5B-Instruct/eval_summary.json"
cat "$ROOT/attack_heavy__Qwen2.5-1.5B-Instruct__full__seed0/eval_final/eval_summary.json"
cat "$ROOT/diverse_attack__Qwen2.5-1.5B-Instruct__full__seed0/eval_final/eval_summary.json"
cat "$ROOT/authorization_balanced__Qwen2.5-1.5B-Instruct__full__seed0/eval_final/eval_summary.json"
```

Per-split metrics include JSON parse rate, exact structured-target accuracy,
action accuracy, authorized and unauthorized exact/action accuracy,
unauthorized-execution rate, counterfactual-pair accuracy, and factor
breakdowns. Predictions are in the corresponding `predictions_<split>.jsonl`.

For checkpoint learning curves:

```bash
python evaluate_checkpoints.py \
  --run-dir "$ROOT/authorization_balanced__Qwen2.5-1.5B-Instruct__full__seed0" \
  --data-dir /local/bys2107/research/auth-training/authorization_dataset_v0/data/generated \
  --hf-cache-dir /local/bys2107/research/auth-training/artifacts/huggingface
```

## W&B-independent visualizations

The repository includes `plot_results.py`, which uses the local JSON artifacts
and does not require a W&B account or network access. Install the added
`matplotlib` dependency with `python -m pip install -r requirements.txt`, then
run it from the local node copy after a smoke or training run:

```bash
RUN=/local/bys2107/research/auth-training/artifacts/runs/authorization_balanced__Qwen2.5-1.5B-Instruct__full__seed0
python plot_results.py --run-dir "$RUN"
```

It writes PNGs to `$RUN/plots/`:

- `training_progress.png`: training loss, held-out validation loss (for full-SFT), learning rate, and gradient norm over optimizer steps;
- `checkpoint_eval_progress.png`: exact-target accuracy, action accuracy, and unauthorized-execution rate across checkpoint evaluations and final;
- `eval_summary.png`: final metrics across the five eval splits.

Choose another output location with `--output-dir`. For the smoke run, use:

```bash
python plot_results.py \
  --run-dir /local/bys2107/research/auth-training/artifacts/runs/smoke_authorization_balanced
```

The plotting command only reads existing logs/metrics and writes small PNG
files; it does not load the model or run evaluation.

## Direct scripts

If already running from the local copy, the underlying commands are:

```bash
bash scripts/smoke_test.sh
bash scripts/eval_baseline.sh
bash scripts/run_full_matrix.sh
```

Useful configuration variables include `ARTIFACT_ROOT`, `OUT_ROOT`,
`HF_CACHE_DIR`, `WANDB_DIR`, `NPROC_PER_NODE`, `MODEL`, `SEED`, and `DATA_DIR`.
The copy helper sets `DATA_DIR` to the absolute local-copy path automatically.
The Python entry points expose equivalent `--output-dir` and
`--hf-cache-dir` options.

## Run artifacts

Each training run writes `run_config.json`, training logs, trainer history,
`train_metrics.json`, periodic checkpoints, and `final/`. Checkpoints include
model weights plus optimizer, scheduler, RNG, and trainer state. Evaluation
writes per-split predictions and metrics plus `eval_summary.json`.

For full SFT, `final/` contains model weights. For LoRA, it contains adapter
weights/configuration and reload metadata; inference reloads the base model
from the configured Hugging Face cache.

## Tests and CPU limitation

Run the non-training checks with:

```bash
python validate_data.py
python -m compileall -q auth_sft train_sft.py evaluate.py validate_data.py evaluate_checkpoints.py
```

The smoke test is the only intentionally trivial training test. Do not run the
full baseline or SFT procedures on a CPU-only workstation; run them inside the
GPU Slurm allocation through `copy_helper.sh`.
