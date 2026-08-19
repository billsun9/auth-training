#!/usr/bin/env python
import argparse
from pathlib import Path

from auth_sft.plotting import (
    plot_checkpoint_eval_progress,
    plot_eval_summary,
    plot_model_comparison,
    plot_training_progress,
)


def plot_run(run_dir: Path, output_dir: Path | None = None):
    """Regenerate all available plots for one local or synced run directory."""
    output_dir = output_dir or run_dir / "plots"
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
        raise FileNotFoundError(f"No plot-ready artifacts under {run_dir}")
    return created


def main():
    p = argparse.ArgumentParser(description="Create W&B-independent plots from experiment reports")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir")
    target.add_argument("--outputs-dir", help="Regenerate plots for every synced run and one model comparison")
    p.add_argument("--output-dir", help="Only valid with --run-dir")
    args = p.parse_args()

    if args.outputs_dir:
        if args.output_dir:
            p.error("--output-dir cannot be used with --outputs-dir")
        root = Path(args.outputs_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Synced outputs directory does not exist: {root}")
        created = []
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            try:
                created.extend(plot_run(run_dir))
            except FileNotFoundError as exc:
                print(f"Skipped: {exc}")
        try:
            created.append(plot_model_comparison(root, root / "model_comparison.png"))
        except FileNotFoundError as exc:
            print(f"Skipped: {exc}")
    else:
        run_dir = Path(args.run_dir)
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
        created = plot_run(run_dir, output_dir)
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
