import json
import hashlib
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

class JSONLLoggingCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            append_jsonl(self.path, {"step": state.global_step, **logs})
