\
#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from authdata.generator import AuthorizationDatasetGenerator, write_jsonl

REGIMES = ["attack_heavy", "diverse_attack", "authorization_balanced"]
EVAL_SPLITS = ["iid", "lexical_ood", "mechanism_ood", "auth_recombination", "benign_control"]
SHARED_EVAL_REGIME = "shared_eval"
DATA_ROOT = Path(__file__).resolve().parent / "data"


def _content_key(row):
    return row.prompt, json.dumps(row.target, sort_keys=True, separators=(",", ":"))


def _count_near_fraction(total, fraction, remainder=1):
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    desired = round(total * fraction)
    candidates = [count for count in range(total + 1) if (total - count) % remainder == 0]
    return min(candidates, key=lambda count: (abs(count - desired), count))


def generate_shared_eval(train_content, n_eval_each, seed, closed_domain_fraction=0.2):
    """Generate one leakage-free shared evaluation suite."""
    eval_seed = seed + 1000003
    while True:
        eval_gen = AuthorizationDatasetGenerator(seed=eval_seed)
        eval_rows = {}
        all_content = set(train_content)
        collision = False
        for split in EVAL_SPLITS:
            remainder = 3 if split == "auth_recombination" else 1
            n_closed = _count_near_fraction(n_eval_each, closed_domain_fraction, remainder)
            rows = eval_gen.generate(SHARED_EVAL_REGIME, split, n_eval_each - n_closed)
            rows.extend(eval_gen.generate_closed_domain(SHARED_EVAL_REGIME, split, n_closed))
            keys = [_content_key(row) for row in rows]
            if any(key in all_content for key in keys):
                collision = True
                break
            all_content.update(keys)
            eval_rows[split] = rows
        if not collision:
            return eval_rows
        eval_seed += 1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=REGIMES)
    p.add_argument("--all-regimes", action="store_true")
    p.add_argument("--preview", action="store_true",
                   help="Replace data/preview with 20 balanced-train rows and 10 rows per eval split")
    p.add_argument("--n-train", type=int, default=2500,
                   help="Regime-specific hierarchy rows per training file")
    p.add_argument("--n-eval-each", type=int, default=100)
    p.add_argument("--n-capability", type=int, default=1000,
                   help="Byte-identical shared capability rows in every train file; 0 disables them")
    p.add_argument("--closed-domain-eval-fraction", type=float, default=0.20,
                   help="Held-out closed-domain task fraction in every shared eval split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.preview and (args.all_regimes or args.regime):
        p.error("--preview cannot be combined with --regime or --all-regimes")
    if not args.preview and not args.all_regimes and not args.regime:
        p.error("choose --regime or --all-regimes")

    if args.preview:
        out = args.out or DATA_ROOT / "preview"
        if out.exists():
            print(f"removing previous preview output: {out}")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        train_rows = AuthorizationDatasetGenerator(seed=args.seed).generate(
            "authorization_balanced", "train", 20,
        )
        write_jsonl(out / "train_20.jsonl", train_rows)
        eval_rows = generate_shared_eval(
            {_content_key(row) for row in train_rows}, 10, args.seed,
            closed_domain_fraction=args.closed_domain_eval_fraction,
        )
        for split, rows in eval_rows.items():
            write_jsonl(out / f"{split}_10.jsonl", rows)
        print(f"generated preview dataset: {out}")
        return

    out = args.out or DATA_ROOT / "generated"

    regimes = REGIMES if args.all_regimes else [args.regime]
    capability_rows = AuthorizationDatasetGenerator(seed=args.seed + 2000003).generate_capability_rehearsal(
        args.n_capability,
    )
    for regime in regimes:
        # A fresh RNG makes each train set depend only on its regime and the
        # requested seed, never on which other regimes were generated first.
        train_gen = AuthorizationDatasetGenerator(seed=args.seed)
        rows = train_gen.generate(regime, "train", args.n_train) + capability_rows
        write_jsonl(out / f"train_{regime}.jsonl", rows)
        print(f"generated train_{regime}: {len(rows)} ({args.n_train} hierarchy + {args.n_capability} shared capability)")

    # Evaluation examples are generated once from a dedicated stream and are
    # rejected/regenerated as a whole if finite-space sampling creates an
    # exact prompt+target duplicate of training or another eval split.
    train_content = set()
    for regime in regimes:
        path = out / f"train_{regime}.jsonl"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            train_content.add((row["prompt"], json.dumps(row["target"], sort_keys=True, separators=(",", ":"))))
    eval_rows = generate_shared_eval(
        train_content, args.n_eval_each, args.seed,
        closed_domain_fraction=args.closed_domain_eval_fraction,
    )
    for split, rows in eval_rows.items():
        write_jsonl(out / f"eval_{split}.jsonl", rows)
    print(f"generated shared eval suite: {len(EVAL_SPLITS) * args.n_eval_each}")

if __name__ == "__main__":
    main()
