#!/usr/bin/env python
import argparse, inspect, json, math, os
from pathlib import Path
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from auth_sft.data import DEFAULT_DATA_DIR, CompletionOnlyCollator, PromptCompletionDataset, canonical_data_paths, load_train_rows, split_train_validation_rows
from auth_sft.logging_utils import JSONLLoggingCallback, write_json
from auth_sft.modeling import load_tokenizer, load_training_model

def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--regime", required=True, choices=["attack_heavy","diverse_attack","authorization_balanced"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--method", choices=["full","lora"], default="full")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-cache-dir", default="artifacts/huggingface",
                   help="Directory for downloaded Hugging Face model/tokenizer weights")
    p.add_argument("--wandb-dir", default="artifacts/wandb",
                   help="Directory for W&B run files when --wandb is enabled")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--validation-ratio", type=float, default=0.1)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--early-stopping-patience", type=int, default=5,
                   help="Validation evaluations without improvement before stopping; 0 disables")
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="auth-training")
    p.add_argument("--run-name")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()

def main():
    a = args_parser()
    set_seed(a.seed)
    canonical_data_paths(a.data_dir)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    if a.wandb:
        os.environ.setdefault("WANDB_PROJECT", a.wandb_project)
        os.environ.setdefault("WANDB_DIR", str(Path(a.wandb_dir)))
    lr = a.learning_rate if a.learning_rate is not None else (2e-5 if a.method == "full" else 1e-4)
    tok = load_tokenizer(a.model, cache_dir=a.hf_cache_dir); tok.padding_side = "right"
    rows = load_train_rows(a.data_dir, a.regime, a.max_train_samples, a.seed)
    train_rows, validation_rows = split_train_validation_rows(rows, a.validation_ratio, a.seed)
    ds = PromptCompletionDataset(train_rows, tok, a.max_seq_length)
    validation_ds = PromptCompletionDataset(validation_rows, tok, a.max_seq_length) if validation_rows else None
    if validation_ds and a.save_steps <= 0:
        raise ValueError("Validation requires checkpoint saving; set --save-steps to a positive value")
    if validation_ds and a.save_steps != a.eval_steps:
        raise ValueError("For validation/best-checkpoint tracking, --save-steps must equal --eval-steps")
    model = load_training_model(a.model, a.method, a.dtype, a.gradient_checkpointing,
                                a.lora_r, a.lora_alpha, a.lora_dropout,
                                cache_dir=a.hf_cache_dir)
    if a.method == "lora" and int(os.environ.get("RANK","0")) == 0:
        model.print_trainable_parameters()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batches = math.ceil(len(ds) / (a.per_device_train_batch_size * world_size))
    updates_per_epoch = max(1, math.ceil(micro_batches / a.gradient_accumulation_steps))
    total_update_steps = a.max_steps if a.max_steps > 0 else math.ceil(updates_per_epoch * a.num_train_epochs)
    resolved_warmup_steps = math.ceil(total_update_steps * a.warmup_ratio)
    training_kwargs = dict(
        output_dir=str(out),
        run_name=a.run_name or f"{a.regime}-{Path(a.model).name}-{a.method}",
        num_train_epochs=a.num_train_epochs, max_steps=a.max_steps,
        learning_rate=lr, weight_decay=a.weight_decay,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        bf16=a.dtype=="bf16", fp16=a.dtype=="fp16", tf32=torch.cuda.is_available(),
        gradient_checkpointing=a.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        logging_strategy="steps", logging_steps=a.logging_steps, logging_first_step=True,
        eval_strategy="steps" if validation_ds else "no",
        eval_steps=a.eval_steps if validation_ds else None,
        save_strategy="steps" if a.save_steps > 0 else "no",
        save_steps=max(1,a.save_steps), save_total_limit=a.save_total_limit,
        load_best_model_at_end=bool(validation_ds),
        metric_for_best_model="eval_loss" if validation_ds else None,
        greater_is_better=False if validation_ds else None,
        report_to=["wandb"] if a.wandb else [],
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False, seed=a.seed, data_seed=a.seed,
        ddp_find_unused_parameters=False,
    )
    # Transformers 5 removed warmup_ratio. Preserve the same ratio for either
    # API so a stale cluster environment fails less surprisingly.
    if "warmup_ratio" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["warmup_ratio"] = a.warmup_ratio
    else:
        training_kwargs["warmup_steps"] = resolved_warmup_steps
    ta = TrainingArguments(**training_kwargs)
    callbacks = [JSONLLoggingCallback(out/"logs"/"train_log.jsonl")]
    if validation_ds and a.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=a.early_stopping_patience))
    trainer = Trainer(
        model=model, args=ta, train_dataset=ds,
        eval_dataset=validation_ds,
        data_collator=CompletionOnlyCollator(tok.pad_token_id),
        callbacks=callbacks,
    )
    cfg = vars(a).copy()
    cfg.update({
        "resolved_learning_rate":lr, "n_train_examples":len(train_rows),
        "n_validation_examples":len(validation_rows),
        "min_token_length":min(ds.lengths), "max_token_length":max(ds.lengths),
        "mean_token_length":sum(ds.lengths)/len(ds.lengths),
        "world_size":world_size,
        "global_train_batch_size": a.per_device_train_batch_size * world_size * a.gradient_accumulation_steps,
        "resolved_warmup_steps": resolved_warmup_steps,
    })
    if trainer.is_world_process_zero(): write_json(out/"run_config.json", cfg)
    result = trainer.train()
    trainer.save_state()
    final = out/"final"
    trainer.save_model(str(final)); tok.save_pretrained(str(final))
    if trainer.is_world_process_zero():
        write_json(out/"train_metrics.json", result.metrics)
        if validation_ds:
            write_json(out/"validation_metrics.json", trainer.evaluate())
        with (out/"logs"/"trainer_log_history.jsonl").open("w",encoding="utf-8") as f:
            for item in trainer.state.log_history:
                f.write(json.dumps(item,ensure_ascii=False)+"\n")
        write_json(final/"auth_training_metadata.json", {
            "base_model":a.model,"method":a.method,"regime":a.regime,"seed":a.seed
        })
        print(f"Saved final model: {final}")

if __name__ == "__main__":
    main()
