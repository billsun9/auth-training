\
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from authdata.generator import AuthorizationDatasetGenerator, write_jsonl

REGIMES = ["attack_heavy", "diverse_attack", "authorization_balanced"]
EVAL_SPLITS = ["iid", "lexical_ood", "mechanism_ood", "auth_recombination", "benign_control"]
SHARED_EVAL_REGIME = "shared_eval"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=REGIMES)
    p.add_argument("--all-regimes", action="store_true")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-eval-each", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data/generated")
    args = p.parse_args()

    if not args.all_regimes and not args.regime:
        p.error("choose --regime or --all-regimes")

    regimes = REGIMES if args.all_regimes else [args.regime]
    for regime in regimes:
        # A fresh RNG makes each train set depend only on its regime and the
        # requested seed, never on which other regimes were generated first.
        train_gen = AuthorizationDatasetGenerator(seed=args.seed)
        write_jsonl(
            args.out / f"train_{regime}.jsonl",
            train_gen.generate(regime, "train", args.n_train),
        )
        print(f"generated train_{regime}: {args.n_train}")

    # Evaluation examples are generated once from a dedicated stream and are
    # rejected/regenerated as a whole if finite-space sampling creates an
    # exact prompt+target duplicate of training or another eval split.
    train_content = set()
    for regime in regimes:
        path = args.out / f"train_{regime}.jsonl"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            train_content.add((row["prompt"], json.dumps(row["target"], sort_keys=True, separators=(",", ":"))))
    eval_seed = args.seed + 1000003
    while True:
        eval_gen = AuthorizationDatasetGenerator(seed=eval_seed)
        eval_rows = {}
        all_content = set(train_content)
        collision = False
        for split in EVAL_SPLITS:
            rows = eval_gen.generate(SHARED_EVAL_REGIME, split, args.n_eval_each)
            keys = [(r.prompt, json.dumps(r.target, sort_keys=True, separators=(",", ":"))) for r in rows]
            if any(k in all_content for k in keys):
                collision = True
                break
            all_content.update(keys)
            eval_rows[split] = rows
        if not collision:
            break
        eval_seed += 1
    for split, rows in eval_rows.items():
        write_jsonl(args.out / f"eval_{split}.jsonl", rows)
    print(f"generated shared eval suite: {len(EVAL_SPLITS) * args.n_eval_each}")

if __name__ == "__main__":
    main()
