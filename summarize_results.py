#!/usr/bin/env python
"""Print a compact comparison table from home-synced experiment reports."""

import argparse
import json
from pathlib import Path


def load_synced_summaries(outputs_dir: str | Path, run_names: set[str] | None = None):
    """Return (run_name, evaluation_kind, summary) triples from synced outputs."""
    root = Path(outputs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Synced outputs directory does not exist: {root}")

    found = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_names and run_dir.name not in run_names:
            continue
        candidates = [
            ("eval_final", run_dir / "eval_final" / "eval_summary.json"),
            ("eval_smoke", run_dir / "eval_smoke" / "eval_summary.json"),
            ("baseline", run_dir / "eval_summary.json"),
        ]
        for evaluation_kind, summary_path in candidates:
            if summary_path.is_file():
                with summary_path.open(encoding="utf-8") as handle:
                    found.append((run_dir.name, evaluation_kind, json.load(handle)))
    if not found:
        requested = f" for {sorted(run_names)}" if run_names else ""
        raise FileNotFoundError(f"No synced eval_summary.json files found under {root}{requested}")
    return found


def _rate(value):
    return "-" if value is None else f"{100 * value:5.1f}%"


def _nested_rate(summary, parent, field):
    return _rate(summary.get(parent, {}).get(field))

def _rate_with_fallback(summary, field, fallback=None):
    value = summary.get(field)
    if value is None and fallback:
        value = summary.get(fallback)
    return _rate(value)


def render_summary_table(records) -> str:
    """Render the primary outcome and authorization metrics for each synced run."""
    lines = []
    header = (
        f"{'split':<28} {'n':>4} {'json':>7} {'exact':>7} {'action':>7} "
        f"{'AER':>7} {'UER':>7} {'auth-ex':>8} {'unauth-ex':>10} {'ref-ex':>7} "
        f"{'pair-ex':>8} {'triplet-ex':>11}"
    )
    for run_name, evaluation_kind, summary in records:
        lines.extend(("", f"{run_name} [{evaluation_kind}]", header))
        for split, metrics in summary.items():
            lines.append(
                f"{split:<28} {metrics.get('n', 0):>4} "
                f"{_rate(metrics.get('json_parse_rate')):>7} "
                f"{_rate(metrics.get('exact_target_accuracy')):>7} "
                f"{_rate(metrics.get('action_accuracy')):>7} "
                f"{_rate_with_fallback(metrics, 'authorized_execution_rate', 'authorized_action_accuracy'):>7} "
                f"{_rate(metrics.get('unauthorized_execution_rate')):>7} "
                f"{_rate(metrics.get('authorized_exact_target_accuracy')):>8} "
                f"{_rate(metrics.get('unauthorized_exact_target_accuracy')):>10} "
                f"{_rate(metrics.get('reference_exact_target_accuracy')):>7} "
                f"{_nested_rate(metrics, 'counterfactual_pairs', 'pair_exact_accuracy'):>8} "
                f"{_nested_rate(metrics, 'counterfactual_triplets', 'triplet_exact_accuracy'):>11}"
            )
            if split == "benign_control":
                lines.append(
                    "  benign: "
                    f"JSON valid={_rate(metrics.get('json_parse_rate'))}, "
                    f"answer action={_rate_with_fallback(metrics, 'benign_answer_action_rate', 'action_accuracy')}, "
                    f"unexpected tool execution={_rate(metrics.get('benign_unexpected_tool_execution_rate'))}, "
                    f"exact content={_rate(metrics.get('exact_target_accuracy'))}"
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
