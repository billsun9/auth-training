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
    fig.suptitle(run_dir.name)
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
    """Plot exact/action/unauthorized-execution metrics across checkpoint evals."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir or run_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_dirs(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint evaluations under {run_dir / 'checkpoint_evals'}")
    splits = sorted(p.stem.removeprefix("metrics_") for p in checkpoints[0][1].glob("metrics_*.json"))
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    x = list(range(len(checkpoints)))
    labels = [p.name for _, p in checkpoints]
    metrics = (
        ("exact_target_accuracy", "exact target accuracy"),
        ("action_accuracy", "action accuracy"),
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
    """Plot final exact/action/unauthorized-execution metrics by eval split."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir or eval_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(eval_dir.glob("metrics_*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics_*.json files under {eval_dir}")
    rows = [(p.stem.removeprefix("metrics_"), _load_json(p)) for p in files]
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    metrics = (
        ("exact_target_accuracy", "exact target"),
        ("action_accuracy", "action"),
        ("unauthorized_execution_rate", "unauthorized execution"),
    )
    x = list(range(len(rows)))
    for axis, (field, title) in zip(axes, metrics):
        values = [metrics_row.get(field) for _, metrics_row in rows]
        axis.bar(x, values)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xticks(x, [name.replace("_", "\n") for name, _ in rows], rotation=0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("accuracy / rate")
    fig.suptitle(eval_dir.name)
    fig.tight_layout()
    out = output_dir / "eval_summary.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
