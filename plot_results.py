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
    # A run has one canonical evaluation suite: eval_final for SFT, eval_smoke
    # for smoke tests, or the run root for a frozen baseline.
    eval_dirs = [run_dir, run_dir / "eval_final", run_dir / "eval_smoke"]
    for eval_dir in eval_dirs:
        if eval_dir.is_dir() and any(eval_dir.glob("metrics_*.json")):
            created.append(plot_eval_summary(eval_dir, output_dir))
    if not created:
        raise SystemExit("No plot-ready artifacts found")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
