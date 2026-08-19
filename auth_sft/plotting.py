"""W&B-independent plots from the JSON artifacts produced by this project."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


SPLIT_ORDER = (
    "iid", "lexical_ood", "mechanism_ood", "auth_recombination",
    "auth_recombination_natural", "authorization_policy_ood", "benign_control",
)


def _ordered_splits(summaries):
    present = {split for summary in summaries for split in summary}
    ordered = [split for split in SPLIT_ORDER if split in present]
    return ordered + sorted(present - set(ordered))


def _split_label(split: str) -> str:
    return split.replace("_", "\n")


def _metric_value(metrics: dict, metric: str):
    """Get a scorecard value, converting lower-is-better safety rates."""
    if metric == "unauthorized_action_avoidance":
        value = metrics.get("unauthorized_execution_rate")
        return None if value is None else 1 - value
    if metric == "benign_no_unexpected_tool_execution":
        value = metrics.get("benign_unexpected_tool_execution_rate")
        return None if value is None else 1 - value
    if metric == "counterfactual_pair_exact":
        return metrics.get("counterfactual_pairs", {}).get("pair_exact_accuracy")
    if metric == "counterfactual_triplet_exact":
        return metrics.get("counterfactual_triplets", {}).get("triplet_exact_accuracy")
    value = metrics.get(metric)
    # Old reports do not contain AER; for these tool-only authorization tasks,
    # authorized action accuracy is its equivalent historical measure.
    if metric == "authorized_execution_rate" and value is None:
        value = metrics.get("authorized_action_accuracy")
    return value


def _score_matrix(rows, columns, metric):
    return [
        [
            value if isinstance(value := _metric_value(row.get(column, {}), metric), (int, float))
            else float("nan")
            for column in columns
        ]
        for row in rows
    ]


def _annotated_heatmap(axis, values, row_labels, column_labels, title):
    """Render an annotation-first rate scorecard; grey cells are not applicable."""
    plt = _plt()
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e5e7eb")
    values = [
        [value if isinstance(value, (int, float)) else float("nan") for value in row]
        for row in values
    ]
    image = axis.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axis.set_title(title, fontsize=11, fontweight="bold", pad=10)
    axis.set_xticks(range(len(column_labels)), column_labels, fontsize=8)
    axis.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            if isinstance(value, (int, float)) and not math.isnan(value):
                label = f"{100 * value:.0f}%"
                color = "white" if value >= 0.58 else "#111827"
            else:
                label, color = "–", "#6b7280"
            axis.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color=color)
    axis.set_xticks([index - 0.5 for index in range(1, len(column_labels))], minor=True)
    axis.set_yticks([index - 0.5 for index in range(1, len(row_labels))], minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    return image


def _run_dir_for_eval(eval_dir: Path) -> Path:
    return eval_dir.parent if eval_dir.name in {"eval_final", "eval_smoke"} else eval_dir


def _run_label(eval_dir: Path) -> str:
    """Return a concise, human-readable model/condition label."""
    run_dir = _run_dir_for_eval(eval_dir)
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        regime = str(config.get("regime", run_dir.name)).replace("_", " ")
        model = str(config.get("model", "unknown model")).split("/")[-1]
        return f"{regime} — {model}"
    config_path = eval_dir / "eval_config.json"
    if config_path.is_file():
        model = str(_load_json(config_path).get("model", "unknown model")).split("/")[-1]
        return f"frozen baseline — {model}"
    return run_dir.name


def plot_training_progress(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Plot logged loss and optimizer diagnostics against training step."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir or run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "logs" / "train_log.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing training log: {path}")
    rows = _load_jsonl(path)
    steps = [r["step"] for r in rows if "step" in r]
    if not steps:
        raise ValueError(f"Training log has no optimizer steps: {path}")

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for field, label in (("loss", "training loss"), ("eval_loss", "validation loss")):
        values = [r.get(field) for r in rows]
        pairs = [(s, v) for s, v in zip(steps, values) if isinstance(v, (int, float))]
        if pairs:
            axes[0].plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=label)
    for field, label in (("learning_rate", "learning rate"), ("grad_norm", "gradient norm")):
        values = [r.get(field) for r in rows]
        pairs = [(s, v) for s, v in zip(steps, values) if isinstance(v, (int, float))]
        if pairs:
            axes[1].plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=label)
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("optimizer diagnostics")
    axes[1].set_xlabel("optimizer step")
    axes[1].legend(loc="best")
    fig.suptitle(f"Training progress — {_run_label(run_dir)}")
    fig.tight_layout()
    out = output_dir / "training_progress.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _checkpoint_dirs(run_dir: Path):
    root = run_dir / "checkpoint_evals"
    dirs = []
    for path in root.iterdir() if root.is_dir() else []:
        if path.is_dir() and (path / "metrics_iid.json").is_file():
            match = re.fullmatch(r"checkpoint-(\d+)", path.name)
            dirs.append((int(match.group(1)) if match else 10**18, path))
    return sorted(dirs, key=lambda item: item[0])


def plot_checkpoint_eval_progress(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Plot exact/action/authorized/unauthorized-execution metrics across checkpoints."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir or run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_dirs(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint evaluations under {run_dir / 'checkpoint_evals'}")
    splits = sorted(p.stem.removeprefix("metrics_") for p in checkpoints[0][1].glob("metrics_*.json"))
    plt = _plt()
    fig, axes = plt.subplots(4, 1, figsize=(11, 15), sharex=True)
    x = list(range(len(checkpoints)))
    labels = [p.name for _, p in checkpoints]
    metrics = (
        ("exact_target_accuracy", "exact target accuracy"),
        ("action_accuracy", "action accuracy"),
        ("authorized_execution_rate", "authorized execution rate"),
        ("unauthorized_execution_rate", "unauthorized execution rate"),
    )
    for axis, (field, title) in zip(axes, metrics):
        for split in splits:
            values = [_load_json(path / f"metrics_{split}.json").get(field) for _, path in checkpoints]
            axis.plot(x, values, marker="o", label=split)
        axis.set_ylabel(title)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=2)
    axes[-1].set_xticks(x, labels, rotation=30, ha="right")
    axes[-1].set_xlabel("checkpoint")
    fig.suptitle(f"Checkpoint evaluation: {run_dir.name}")
    fig.tight_layout()
    out = output_dir / "checkpoint_eval_progress.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_eval_summary(eval_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Render an annotated final-evaluation scorecard for one model run."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir or eval_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(eval_dir.glob("metrics_*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics_*.json files under {eval_dir}")
    summary = {p.stem.removeprefix("metrics_"): _load_json(p) for p in files}
    splits = _ordered_splits([summary])
    plt = _plt()
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 12),
        gridspec_kw={"height_ratios": [10, 3]}, layout="constrained",
    )
    primary_metrics = (
        ("json_parse_rate", "JSON valid"),
        ("exact_target_accuracy", "Exact target"),
        ("action_accuracy", "Action accuracy"),
        ("authorized_execution_rate", "Authorized execution (AER)"),
        ("unauthorized_action_avoidance", "Unauthorized action avoidance"),
        ("authorized_exact_target_accuracy", "Authorized exact"),
        ("unauthorized_exact_target_accuracy", "Unauthorized exact"),
        ("reference_exact_target_accuracy", "Reference exact"),
        ("counterfactual_pair_exact", "Counterfactual pair exact"),
        ("counterfactual_triplet_exact", "Counterfactual triplet exact"),
    )
    _annotated_heatmap(
        axes[0],
        [_score_matrix([summary], splits, metric)[0] for metric, _ in primary_metrics],
        [label for _, label in primary_metrics], [_split_label(split) for split in splits],
        "Evaluation scorecard (higher is better; grey = not applicable)",
    )
    benign_metrics = (
        ("json_parse_rate", "JSON valid"),
        ("benign_answer_action_rate", "Answer action"),
        ("benign_no_unexpected_tool_execution", "No unexpected tool action"),
        ("exact_target_accuracy", "Exact content"),
    )
    _annotated_heatmap(
        axes[1],
        [[_metric_value(summary.get("benign_control", {}), metric) for metric, _ in benign_metrics]],
        ["benign control"], [label for _, label in benign_metrics],
        "Benign-control behavior",
    )
    fig.suptitle(f"Final evaluation — {_run_label(eval_dir)}")
    out = output_dir / "eval_summary.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _comparison_label(run_dir: Path) -> str | None:
    name = run_dir.name
    condition = None
    if name.startswith("baseline__"):
        condition = "Frozen baseline"
    for prefix, label in (
        ("capability_only__", "Capability-only SFT"),
        ("attack_heavy__", "Attack-heavy SFT"),
        ("diverse_attack__", "Diverse-attack SFT"),
        ("authorization_balanced__", "Authorization-balanced SFT"),
    ):
        if name.startswith(prefix):
            condition = label
            break
    model = "unknown model"
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        model = str(_load_json(config_path).get("model", model)).split("/")[-1]
    elif (run_dir / "eval_config.json").is_file():
        model = str(_load_json(run_dir / "eval_config.json").get("model", model)).split("/")[-1]
    return f"{condition}\n{model}" if condition else None


def _comparison_order(label: str) -> int:
    condition = label.split("\n", 1)[0]
    return {
        "Frozen baseline": 0,
        "Capability-only SFT": 1,
        "Attack-heavy SFT": 2,
        "Diverse-attack SFT": 3,
        "Authorization-balanced SFT": 4,
    }.get(condition, 99)


def plot_model_comparison(runs_root: str | Path, output_path: str | Path) -> Path:
    """Compare final evaluation metrics across every available model condition."""
    runs_root = Path(runs_root)
    records = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        summary_path = run_dir / "eval_final" / "eval_summary.json"
        if not summary_path.is_file():
            summary_path = run_dir / "eval_summary.json"
        label = _comparison_label(run_dir)
        if label and summary_path.is_file():
            records.append((label, _load_json(summary_path)))
    if not records:
        raise FileNotFoundError(f"No final eval_summary.json files under {runs_root}")
    records.sort(key=lambda item: (_comparison_order(item[0]), item[0]))

    labels = [label for label, _ in records]
    summaries = [summary for _, summary in records]
    splits = _ordered_splits(summaries)
    plt = _plt()
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), layout="constrained")
    axes = axes.ravel()
    scorecards = (
        ("exact_target_accuracy", "Exact target accuracy"),
        ("json_parse_rate", "JSON validity"),
        ("authorized_execution_rate", "Authorized execution rate (AER)"),
        ("unauthorized_action_avoidance", "Unauthorized action avoidance (1 − UER)"),
    )
    for axis, (metric, title) in zip(axes, scorecards):
        _annotated_heatmap(
            axis, _score_matrix(summaries, splits, metric), labels,
            [_split_label(split) for split in splits], title,
        )
    recombination_splits = [
        split for split in (
            "auth_recombination", "auth_recombination_natural", "authorization_policy_ood",
        ) if split in splits
    ]
    if not recombination_splits:
        recombination_splits = ["not_available"]
    _annotated_heatmap(
        axes[4], _score_matrix(summaries, recombination_splits, "counterfactual_triplet_exact"), labels,
        ["not\navailable" if split == "not_available" else _split_label(split) for split in recombination_splits],
        "Counterfactual triplet exact accuracy",
    )
    benign_metrics = (
        ("json_parse_rate", "JSON valid"),
        ("benign_answer_action_rate", "Answer action"),
        ("benign_no_unexpected_tool_execution", "No unexpected tool action"),
        ("exact_target_accuracy", "Exact content"),
    )
    benign_values = [
        [_metric_value(summary.get("benign_control", {}), metric) for metric, _ in benign_metrics]
        for summary in summaries
    ]
    _annotated_heatmap(
        axes[5], benign_values, labels, [label for _, label in benign_metrics],
        "Benign-control behavior",
    )
    fig.suptitle("Final evaluation comparison across available model conditions")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
