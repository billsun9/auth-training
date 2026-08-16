#!/usr/bin/env python
import argparse
from collections import Counter
import json
from auth_sft.data import TRAIN_FILES,EVAL_FILES,canonical_data_paths,load_train_rows,load_eval_rows

def content_key(row):
    return (row["prompt"], json.dumps(row["target"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))

def summarize(rows):
    return {
        "n":len(rows),
        "authorized":dict(Counter(r["authorized"] for r in rows)),
        "source":dict(Counter(r["source"] for r in rows)),
        "style":dict(Counter(r["style"] for r in rows)),
        "mechanism":dict(Counter(r["mechanism"] for r in rows)),
        "candidate_action":dict(Counter(r["candidate_action"] for r in rows)),
    }

def assert_unique_ids(rows, label):
    ids=[r["id"] for r in rows]
    dup=[x for x,n in Counter(ids).items() if n>1]
    if dup:
        raise ValueError(f"Duplicate IDs within {label}, e.g. {dup[:3]}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="authorization_dataset_v0/data/generated")
    a=p.parse_args()
    canonical_data_paths(a.data_dir)

    # Regimes are allowed to share training examples. That is not leakage:
    # they are separate experimental conditions, not train/test partitions.
    all_train_ids=set()
    all_train_content=set()
    for regime in TRAIN_FILES:
        rows=load_train_rows(a.data_dir,regime)
        assert_unique_ids(rows,f"train/{regime}")
        all_train_ids.update(r["id"] for r in rows)
        all_train_content.update(content_key(r) for r in rows)
        print("train",regime,summarize(rows))

    all_eval_ids=set()
    for split in EVAL_FILES:
        rows=load_eval_rows(a.data_dir,split)
        assert_unique_ids(rows,f"eval/{split}")
        ids={r["id"] for r in rows}
        if all_train_ids & ids:
            raise ValueError(f"Train/eval ID leakage in {split}")
        overlap = all_train_content & {content_key(r) for r in rows}
        if overlap:
            raise ValueError(f"Train/eval prompt-target leakage in {split}: {len(overlap)} exact duplicates")
        if all_eval_ids & ids:
            raise ValueError(f"Duplicate IDs across eval splits involving {split}")
        all_eval_ids |= ids
        print("eval",split,summarize(rows))

    print("Basic schema + canonical-layout + train/eval ID-disjointness validation passed.")

if __name__=="__main__":
    main()
