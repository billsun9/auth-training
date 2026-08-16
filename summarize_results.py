#!/usr/bin/env python
"""Print a compact comparison table from home-synced experiment reports."""

import argparse
import json
from pathlib import Path


SUMMARY_CANDIDATES = (
    ("eval_final", "eval_summary.json"),
    ("eval_smoke", "eval_summary.json"),
    ("", "eval_summary.json"),
)


def load_synced_summaries(outputs_dir: str | Path, run_names: set[str] | None = None):
    """Return (run_name, evaluation_kind, summary) triples from synced outputs."""
    root = Path(outputs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Synced outputs directory does not exist: {root}")

    found = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_names and run_dir.name not in run_names:
            continue
        for parent, filename in SUMMARY_CANDIDATES:
            summary_path = run_dir / parent / filename if parent else run_dir / filename
            if summary_path.is_file():
                with summary_path.open(encoding="utf-8") as handle:
                    found.append((run_dir.name, parent or "baseline", json.load(handle)))
                break
    if not found:
        requested = f" for {sorted(run_names)}" if run_names else ""
        raise FileNotFoundError(f"No synced eval_summary.json files found under {root}{requested}")
    return found


def _rate(value):
    return "-" if value is None else f"{100 * value:5.1f}%"


def _nested_rate(summary, parent, field):
    return _rate(summary.get(parent, {}).get(field))


def render_summary_table(records) -> str:
    """Render the primary outcome and authorization metrics for each synced run."""
    lines = []
    header = (
        f"{'split':<20} {'n':>4} {'json':>7} {'exact':>7} {'action':>7} "
        f"{'auth-ex':>8} {'unauth-ex':>10} {'UER':>7} {'ref-ex':>7} "
        f"{'pair-ex':>8} {'triplet-ex':>11}"
    )
    for run_name, evaluation_kind, summary in records:
        lines.extend(("", f"{run_name} [{evaluation_kind}]", header))
        for split, metrics in summary.items():
            lines.append(
                f"{split:<20} {metrics.get('n', 0):>4} "
                f"{_rate(metrics.get('json_parse_rate')):>7} "
                f"{_rate(metrics.get('exact_target_accuracy')):>7} "
                f"{_rate(metrics.get('action_accuracy')):>7} "
                f"{_rate(metrics.get('authorized_exact_target_accuracy')):>8} "
                f"{_rate(metrics.get('unauthorized_exact_target_accuracy')):>10} "
                f"{_rate(metrics.get('unauthorized_execution_rate')):>7} "
                f"{_rate(metrics.get('reference_exact_target_accuracy')):>7} "
                f"{_nested_rate(metrics, 'counterfactual_pairs', 'pair_exact_accuracy'):>8} "
                f"{_nested_rate(metrics, 'counterfactual_triplets', 'triplet_exact_accuracy'):>11}"
            )
    return "\n".join(lines).lstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Print key metrics from reports synced to the home checkout."
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--runs", nargs="+", help="Optional exact run-directory names to include")
    args = parser.parse_args()
    records = load_synced_summaries(args.outputs_dir, set(args.runs or []))
    print(render_summary_table(records), end="")


if __name__ == "__main__":
    main()
