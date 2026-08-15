\
#!/usr/bin/env python3
import argparse
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
    p.add_argument("--out", type=Path, default=Path("data/generated"))
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

    # Evaluation examples are generated once from a dedicated generator.  They
    # therefore remain byte-identical across all training-regime comparisons.
    eval_gen = AuthorizationDatasetGenerator(seed=args.seed)
    for split in EVAL_SPLITS:
        write_jsonl(
            args.out / f"eval_{split}.jsonl",
            eval_gen.generate(SHARED_EVAL_REGIME, split, args.n_eval_each),
        )
    print(f"generated shared eval suite: {len(EVAL_SPLITS) * args.n_eval_each}")

if __name__ == "__main__":
    main()
