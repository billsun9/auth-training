"""W&B-independent plots from the JSON artifacts produced by this project."""

from __future__ import annotations

import json
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
    """Plot final exact/action/authorized/unauthorized-execution metrics by eval split."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir or eval_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(eval_dir.glob("metrics_*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics_*.json files under {eval_dir}")
    rows = [(p.stem.removeprefix("metrics_"), _load_json(p)) for p in files]
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.ravel()
    metrics = (
        ("exact_target_accuracy", "exact target"),
        ("action_accuracy", "action"),
        ("authorized_execution_rate", "authorized execution"),
        ("unauthorized_execution_rate", "unauthorized execution"),
    )
    y = list(range(len(rows)))
    for axis, (field, title) in zip(axes, metrics):
        values = [
            value if isinstance(value, (int, float)) else float("nan")
            for value in (metrics_row.get(field) for _, metrics_row in rows)
        ]
        axis.barh(y, values)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xlim(0, 1)
        axis.set_yticks(y, [name.replace("_", " ") for name, _ in rows])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Final evaluation — {_run_label(eval_dir)}")
    fig.tight_layout()
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

    splits = []
    for _, summary in records:
        for split in summary:
            if split not in splits:
                splits.append(split)
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    axes = axes.ravel()
    metrics = (
        ("exact_target_accuracy", "Exact target accuracy"),
        ("action_accuracy", "Action accuracy"),
        ("authorized_execution_rate", "Authorized execution rate (AER)"),
        ("unauthorized_execution_rate", "Unauthorized execution rate (UER; lower is better)"),
    )
    height = 0.78 / len(records)
    centers = list(range(len(splits)))
    for axis, (field, title) in zip(axes, metrics):
        for index, (label, summary) in enumerate(records):
            values = []
            for split in splits:
                split_metrics = summary.get(split, {})
                value = split_metrics.get(field)
                # Older reports predate the explicit AER field; their
                # authorized-action metric is equivalent for these tool-only
                # authorization examples.
                if field == "authorized_execution_rate" and value is None:
                    value = split_metrics.get("authorized_action_accuracy")
                values.append(value)
            values = [value if isinstance(value, (int, float)) else float("nan") for value in values]
            positions = [center - 0.39 + height * (index + 0.5) for center in centers]
            axis.barh(positions, values, height=height, label=label)
        axis.set_title(title)
        axis.set_xlim(0, 1)
        axis.set_yticks(centers, [split.replace("_", " ") for split in splits])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(records)))
    fig.suptitle("Final evaluation comparison across available model conditions")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
