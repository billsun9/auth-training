#!/usr/bin/env python
import argparse
from pathlib import Path

from auth_sft.plotting import (
    plot_checkpoint_eval_progress,
    plot_eval_summary,
    plot_training_progress,
)


def main():
    p = argparse.ArgumentParser(description="Create W&B-independent plots from an auth-training run")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir")
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for fn in (
        lambda: plot_training_progress(run_dir, output_dir),
        lambda: plot_checkpoint_eval_progress(run_dir, output_dir),
    ):
        try:
            created.append(fn())
        except FileNotFoundError as exc:
            print(f"Skipped: {exc}")
    eval_dirs = [run_dir]
    eval_dirs.extend(sorted(path for path in run_dir.glob("eval_*") if path.is_dir()))
    for eval_dir in eval_dirs:
        if eval_dir.is_dir() and any(eval_dir.glob("metrics_*.json")):
            target_dir = output_dir if eval_dir == run_dir or eval_dir.name in {
                "eval_final", "eval_smoke"
            } else output_dir / eval_dir.name
            created.append(plot_eval_summary(eval_dir, target_dir))
    if not created:
        raise SystemExit("No plot-ready artifacts found")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
