#!/usr/bin/env python
import argparse, json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from auth_sft.data import EVAL_FILES, canonical_data_paths, load_eval_rows, prompt_with_separator
from auth_sft.logging_utils import sha256_file, write_json
from auth_sft.metrics import compute_metrics, parse_json_object
from auth_sft.modeling import load_inference_model

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="authorization_dataset_v0/data/generated")
    p.add_argument("--model",required=True,help="HF model name or local final/checkpoint directory")
    p.add_argument("--tokenizer",help="Optional tokenizer name/path; useful for model-only checkpoints")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--hf-cache-dir",default="artifacts/huggingface",
                   help="Directory for downloaded Hugging Face model/tokenizer weights")
    p.add_argument("--splits",nargs="+",default=["all"],choices=["all",*EVAL_FILES.keys()])
    p.add_argument("--max-samples-per-split",type=int)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-new-tokens",type=int,default=128)
    p.add_argument("--dtype",choices=["bf16","fp16","fp32"],default="bf16")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--append", action="store_true",
                   help="Add these splits to an existing evaluation directory without replacing its other split results")
    return p.parse_args()

@torch.inference_mode()
def generate_predictions(model,tok,rows,batch_size,max_new_tokens):
    tok.padding_side="left"; raws=[]; preds=[]; device=model.device
    model.generation_config.do_sample = False
    for field in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, field, None)
    for start in tqdm(range(0,len(rows),batch_size),desc="generate",leave=False):
        batch=rows[start:start+batch_size]
        prompts=[prompt_with_separator(r["prompt"]) for r in batch]
        enc=tok(prompts,return_tensors="pt",padding=True,truncation=False,add_special_tokens=False)
        enc={k:v.to(device) for k,v in enc.items()}
        width=enc["input_ids"].shape[1]
        out=model.generate(**enc,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
                           eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
        texts=tok.batch_decode(out[:,width:],skip_special_tokens=True)
        raws.extend(texts); preds.extend(parse_json_object(t) for t in texts)
    return raws,preds

def main():
    a=parse_args(); canonical_data_paths(a.data_dir)
    splits=list(EVAL_FILES) if "all" in a.splits else a.splits
    model,tok=load_inference_model(
        a.model,a.dtype,cache_dir=a.hf_cache_dir,tokenizer_name_or_path=a.tokenizer,
    )
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    config_path = out / "eval_config.json"
    summary_path = out / "eval_summary.json"
    summary = {}
    config = {
        "model": a.model, "tokenizer": a.tokenizer or a.model,
        "dtype": a.dtype, "batch_size": a.batch_size,
        "max_new_tokens": a.max_new_tokens, "seed": a.seed,
        "max_samples_per_split": a.max_samples_per_split,
        "split_sha256": {
            split: sha256_file(Path(a.data_dir) / EVAL_FILES[split]) for split in splits
        },
    }
    if a.append and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if a.append and config_path.is_file():
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        for field in ("model", "tokenizer", "dtype", "seed", "max_samples_per_split"):
            if prior_config.get(field) != config[field]:
                raise ValueError(
                    f"Refusing to append results with a different {field}: "
                    f"existing={prior_config.get(field)!r}, requested={config[field]!r}"
                )
        config["split_sha256"] = {**prior_config.get("split_sha256", {}), **config["split_sha256"]}
    write_json(config_path, config)
    for split in splits:
        rows=load_eval_rows(a.data_dir,split,a.max_samples_per_split,a.seed)
        raws,preds=generate_predictions(model,tok,rows,a.batch_size,a.max_new_tokens)
        metrics=compute_metrics(rows,raws,preds); summary[split]=metrics
        with (out/f"predictions_{split}.jsonl").open("w",encoding="utf-8") as f:
            for r,text,pred in zip(rows,raws,preds):
                f.write(json.dumps({
                    "id":r["id"],"split":split,"authorized":r["authorized"],
                    "candidate_action":r["candidate_action"],"target":r["target"],
                    "prediction":pred,"raw_completion":text,"exact":pred==r["target"],
                    "metadata":r["metadata"],
                },ensure_ascii=False)+"\n")
        write_json(out/f"metrics_{split}.json",metrics)
        print(split, "exact=",metrics["exact_target_accuracy"],
              "action=",metrics["action_accuracy"],
              "unauth_exec=",metrics["unauthorized_execution_rate"])
    write_json(out/"eval_summary.json",summary)
    print(f"Wrote: {out/'eval_summary.json'}")

if __name__=="__main__":
    main()
