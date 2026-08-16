#!/usr/bin/env python
import argparse,re,subprocess,sys
from pathlib import Path
def step(p):
    m=re.fullmatch(r"checkpoint-(\d+)",p.name)
    return int(m.group(1)) if m else 10**18
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",required=True); p.add_argument("--data-dir",default="authorization_dataset_v0/data/generated")
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--max-samples-per-split",type=int)
    p.add_argument("--dtype",default="bf16"); p.add_argument("--hf-cache-dir",default="artifacts/huggingface"); a=p.parse_args()
    run=Path(a.run_dir); ckpts=sorted(run.glob("checkpoint-*"),key=step)
    if (run/"final").is_dir(): ckpts.append(run/"final")
    if not ckpts: raise SystemExit(f"No checkpoints under {run}")
    for ck in ckpts:
        cmd=[sys.executable,"evaluate.py","--data-dir",a.data_dir,"--model",str(ck),
             "--tokenizer",str(run/"final"),
             "--output-dir",str(run/"checkpoint_evals"/ck.name),"--batch-size",str(a.batch_size),
             "--dtype",a.dtype,"--hf-cache-dir",a.hf_cache_dir,"--splits","all"]
        if a.max_samples_per_split is not None:
            cmd += ["--max-samples-per-split",str(a.max_samples_per_split)]
        print("+"," ".join(cmd)); subprocess.run(cmd,check=True)
if __name__=="__main__": main()
