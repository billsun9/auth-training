import json
import hashlib
import shutil
from pathlib import Path
from transformers import TrainerCallback

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def prune_periodic_checkpoints(run_dir):
    """Remove only direct Trainer checkpoint directories after final export."""
    removed = []
    for path in Path(run_dir).glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path.name)
    return sorted(removed)

class JSONLLoggingCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            append_jsonl(self.path, {"step": state.global_step, **logs})
