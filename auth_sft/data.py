from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

TRAIN_FILES = {
    "attack_heavy": "train_attack_heavy.jsonl",
    "diverse_attack": "train_diverse_attack.jsonl",
    "authorization_balanced": "train_authorization_balanced.jsonl",
}
EVAL_FILES = {
    "iid": "eval_iid.jsonl",
    "lexical_ood": "eval_lexical_ood.jsonl",
    "mechanism_ood": "eval_mechanism_ood.jsonl",
    "auth_recombination": "eval_auth_recombination.jsonl",
    "benign_control": "eval_benign_control.jsonl",
}
REQUIRED_KEYS = {
    "id","split","regime","source","authorized","style","mechanism","domain",
    "candidate_action","prompt","target","metadata",
}

DEFAULT_DATA_DIR = "authorization_dataset_v0/data/generated"

def prompt_with_separator(prompt: str) -> str:
    """Preserve the rendered prompt and append one canonical separator."""
    return prompt + ("" if prompt.endswith("\n") else "\n")

def canonical_target(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))

def validate_row(row: dict[str, Any], path=None, line_no=None) -> None:
    where = ""
    if path is not None:
        where = str(path) + (f":{line_no}" if line_no is not None else "") + ": "
    missing = REQUIRED_KEYS - row.keys()
    if missing:
        raise ValueError(f"{where}missing required keys: {sorted(missing)}")
    if not isinstance(row["id"], str):
        raise ValueError(f"{where}id must be a string")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise ValueError(f"{where}prompt must be a non-empty string")
    if not isinstance(row["target"], dict) or "action" not in row["target"]:
        raise ValueError(f"{where}target must be an object containing 'action'")
    if not isinstance(row["authorized"], bool):
        raise ValueError(f"{where}authorized must be boolean")
    if not isinstance(row["candidate_action"], str):
        raise ValueError(f"{where}candidate_action must be a string")
    if not isinstance(row["metadata"], dict):
        raise ValueError(f"{where}metadata must be an object")

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            validate_row(row, path, line_no)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no examples")
    return rows

def canonical_data_paths(data_dir: str | Path) -> dict[str, Path]:
    data_dir = Path(data_dir)
    paths = {}
    for regime, filename in TRAIN_FILES.items():
        paths[f"train:{regime}"] = data_dir / filename
    for split, filename in EVAL_FILES.items():
        paths[f"eval:{split}"] = data_dir / filename
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Expected ONLY the canonical shared-eval dataset layout. Missing:\n  "
            + "\n  ".join(missing)
        )
    return paths

def deterministic_subset(rows, max_samples, seed):
    if max_samples is None or max_samples >= len(rows):
        return rows
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in sorted(idx[:max_samples])]

def load_train_rows(data_dir, regime, max_samples=None, seed=0):
    if regime not in TRAIN_FILES:
        raise ValueError(f"Unknown regime: {regime}")
    canonical_data_paths(data_dir)
    rows = read_jsonl(Path(data_dir) / TRAIN_FILES[regime])
    if any(r["split"] != "train" for r in rows):
        raise ValueError(f"{TRAIN_FILES[regime]} contains a non-train split")
    if any(r["regime"] != regime for r in rows):
        raise ValueError(f"{TRAIN_FILES[regime]} contains a mismatched regime")
    return deterministic_subset(rows, max_samples, seed)

def load_eval_rows(data_dir, split, max_samples=None, seed=0):
    if split not in EVAL_FILES:
        raise ValueError(f"Unknown eval split: {split}")
    canonical_data_paths(data_dir)
    rows = read_jsonl(Path(data_dir) / EVAL_FILES[split])
    if any(r["split"] != split for r in rows):
        raise ValueError(f"{EVAL_FILES[split]} contains a mismatched split")
    if any(r["regime"] != "shared_eval" for r in rows):
        raise ValueError(f"{EVAL_FILES[split]} should use regime='shared_eval'")
    return deterministic_subset(rows, max_samples, seed)

class PromptCompletionDataset(Dataset):
    """
    Causal-LM SFT dataset with completion-only loss.
    Prompt tokens are masked to -100; only the target JSON contributes loss.
    """
    def __init__(self, rows, tokenizer, max_length=1024, allow_truncation=False):
        if allow_truncation:
            raise ValueError("Truncation is disabled; increase max_length to preserve the full prompt and target.")
        self.examples = []
        self.lengths = []
        for row in rows:
            prompt_text = prompt_with_separator(row["prompt"])
            completion_text = canonical_target(row["target"])
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
            completion_ids = tokenizer(completion_text, add_special_tokens=False, truncation=False)["input_ids"]
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define EOS for completion-only training")
            completion_ids = completion_ids + [tokenizer.eos_token_id]
            total = len(prompt_ids) + len(completion_ids)
            if total > max_length:
                raise ValueError(
                    f"Example {row['id']} is {total} tokens > max_length={max_length}. "
                    "Increase --max-seq-length; truncation would change the protocol."
                )
            input_ids = prompt_ids + completion_ids
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor([-100] * len(prompt_ids) + completion_ids, dtype=torch.long),
            })
            self.lengths.append(len(input_ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(x["input_ids"].shape[0] for x in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            pad = max_len - x["input_ids"].shape[0]
            out["input_ids"].append(torch.cat([
                x["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)
            ]))
            out["attention_mask"].append(torch.cat([
                x["attention_mask"], torch.zeros(pad, dtype=torch.long)
            ]))
            out["labels"].append(torch.cat([
                x["labels"], torch.full((pad,), -100, dtype=torch.long)
            ]))
        return {k: torch.stack(v) for k, v in out.items()}
