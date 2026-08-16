from __future__ import annotations
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def dtype_from_name(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]

def load_tokenizer(name, cache_dir=None):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True, cache_dir=cache_dir)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token for completion-only SFT and greedy eval")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok

def load_training_model(model_name, method, dtype, gradient_checkpointing,
                        lora_r=16, lora_alpha=32, lora_dropout=0.05,
                        cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype_from_name(dtype), low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    model.config.use_cache = not gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if method == "full":
        return model
    if method != "lora":
        raise ValueError("method must be full or lora")
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
    )
    return get_peft_model(model, cfg)

def load_inference_model(model_name_or_path, dtype="bf16", cache_dir=None):
    path = Path(model_name_or_path)
    if (path / "adapter_config.json").is_file():
        from peft import PeftConfig, PeftModel
        cfg = PeftConfig.from_pretrained(model_name_or_path)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name_or_path, torch_dtype=dtype_from_name(dtype),
            device_map="auto", low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        model = PeftModel.from_pretrained(base, model_name_or_path)
        tokenizer = load_tokenizer(model_name_or_path, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype_from_name(dtype),
            device_map="auto", low_cpu_mem_usage=True, cache_dir=cache_dir
        )
        tokenizer = load_tokenizer(model_name_or_path, cache_dir=cache_dir)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer
