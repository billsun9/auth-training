#!/usr/bin/env python
"""Local browser for generated authorization data and synced evaluation reports."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from html import escape
from pathlib import Path

DATA_FILES = (
    "train_attack_heavy.jsonl",
    "train_diverse_attack.jsonl",
    "train_authorization_balanced.jsonl",
    "train_capability_only.jsonl",
    "eval_iid.jsonl",
    "eval_lexical_ood.jsonl",
    "eval_mechanism_ood.jsonl",
    "eval_auth_recombination.jsonl",
    "eval_auth_recombination_natural.jsonl",
    "eval_authorization_policy_ood.jsonl",
    "eval_benign_control.jsonl",
)

EVAL_FILE_BY_SPLIT = {
    "iid": "eval_iid.jsonl",
    "lexical_ood": "eval_lexical_ood.jsonl",
    "mechanism_ood": "eval_mechanism_ood.jsonl",
    "auth_recombination": "eval_auth_recombination.jsonl",
    "auth_recombination_natural": "eval_auth_recombination_natural.jsonl",
    "authorization_policy_ood": "eval_authorization_policy_ood.jsonl",
    "benign_control": "eval_benign_control.jsonl",
}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorization experiment browser</title>
<style>
:root { --ink:#172033; --muted:#60708a; --line:#d8e0ec; --bg:#f5f7fb; --card:#fff; --system:#ece9ff; --user:#dff4ff; --external:#fff2d8; --truth:#e3f8ec; --prediction:#ffe5e7; --good:#087443; --bad:#a92638; }
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}.shell{display:grid;grid-template-columns:300px minmax(0,1fr);min-height:100vh}.sidebar{background:#10192b;color:#eaf0ff;padding:22px 16px;position:sticky;top:0;height:100vh;overflow:auto}.brand{font-size:18px;font-weight:750;letter-spacing:-.02em;margin-bottom:5px}.subtitle{color:#9eb0d2;font-size:12px;margin-bottom:23px}.section{margin:20px 0 8px;color:#9eb0d2;font-size:11px;font-weight:750;letter-spacing:.08em;text-transform:uppercase}.nav-link{display:block;color:#dce7ff;text-decoration:none;padding:8px 9px;border-radius:7px;overflow-wrap:anywhere}.nav-link:hover,.nav-link.active{background:#253554;color:#fff}.run{margin:9px 0}.run-name{color:#b8c8e7;font-size:11px;padding:4px 9px;overflow-wrap:anywhere}.main{max-width:920px;width:100%;margin:0 auto;padding:32px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#5870a1;font-weight:800}.headline{font-size:27px;letter-spacing:-.035em;margin:4px 0 6px}.meta{color:var(--muted);margin-bottom:22px}.notice{background:#fff6d9;border:1px solid #f2df97;padding:12px 14px;border-radius:9px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:17px;margin:14px 0;box-shadow:0 1px 2px #16213b08}.panel-stack{display:grid;gap:12px}.panel{border-radius:10px;padding:15px;border:1px solid var(--line)}.panel.system{background:var(--system)}.panel.user{background:var(--user)}.panel.external{background:var(--external)}.panel h3{margin:0 0 7px;font-size:12px;text-transform:uppercase;letter-spacing:.06em}.panel pre,.json pre,.raw-io pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.truth{background:var(--truth);border-color:#a8e1bc}.prediction{background:var(--prediction);border-color:#f2b4bb}.status{display:inline-block;font-weight:800;padding:3px 8px;border-radius:99px;font-size:12px}.status.good{color:var(--good);background:#d9f5e5}.status.bad{color:var(--bad);background:#ffe1e5}.pager{display:flex;align-items:center;justify-content:space-between;gap:10px}.button{display:inline-block;text-decoration:none;color:#fff;background:#2456b3;padding:8px 12px;border-radius:8px;font-weight:700}.button.disabled{pointer-events:none;background:#aab6cb}.facts{display:flex;flex-wrap:wrap;gap:7px}.fact{background:#eef2f8;border-radius:99px;padding:3px 8px;color:#485875;font-size:12px}.raw-io summary{cursor:pointer;color:#2456b3;font-weight:800}.raw-io pre{margin-top:12px;background:#10192b;color:#eaf0ff;padding:14px;border-radius:8px}.token-note{color:var(--muted);font-size:12px;margin:8px 0 0}.empty{color:var(--muted);font-style:italic}@media(max-width:850px){.shell{display:block}.sidebar{position:static;height:auto}.main{padding:20px}}
</style></head><body><div class="shell">
<aside class="sidebar"><div class="brand">Authorization browser</div><div class="subtitle">Generated data + synced predictions</div>
<a class="nav-link {% if view == 'home' %}active{% endif %}" href="{{ url_for('home') }}">Overview</a>
<div class="section">Generated data</div>{% for file in data_files %}<a class="nav-link {% if selected == file %}active{% endif %}" href="{{ url_for('dataset', filename=file) }}">{{ file }}</a>{% endfor %}
<div class="section">Synced predictions</div>{% if prediction_index %}{% for run, splits in prediction_index.items() %}<div class="run"><div class="run-name">{{ run }}</div>{% for split in splits %}<a class="nav-link {% if selected_prediction == (run, split) %}active{% endif %}" href="{{ url_for('prediction', run=run, split=split) }}">{{ split }}</a>{% endfor %}</div>{% endfor %}{% else %}<div class="subtitle">No predictions found. Run sync first.</div>{% endif %}
</aside><main class="main">{{ body|safe }}</main></div></body></html>"""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_components(prompt: str) -> list[tuple[str, str, str]]:
    """Split the fixed prompt convention into visually distinct roles."""
    system, remainder = prompt, ""
    if "\n\nUSER:\n" in prompt:
        system, remainder = prompt.split("\n\nUSER:\n", 1)
    chunks = remainder.split("\n\n") if remainder else []
    components = [("System instruction", "system", system)]
    if chunks:
        components.append(("User request", "user", chunks[0]))
    if len(chunks) > 1:
        components.append(("External content", "external", "\n\n".join(chunks[1:])))
    return [(title, style, text) for title, style, text in components if text]


def json_block(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) if value is not None else "null"


def page_index(requested: str | None, total: int) -> int:
    try:
        index = int(requested or 0)
    except ValueError:
        index = 0
    return min(max(index, 0), max(total - 1, 0))


def create_app(data_dir: str | Path, outputs_dir: str | Path) -> "Flask":
    try:
        from flask import Flask, abort, render_template_string, request, url_for
    except ImportError as exc:
        raise RuntimeError("Flask is required; run: python -m pip install -r requirements.txt") from exc
    data_root, output_root = Path(data_dir), Path(outputs_dir)
    app = Flask(__name__)

    @lru_cache(maxsize=None)
    def rows(path: str):
        return read_jsonl(Path(path))

    @lru_cache(maxsize=None)
    def canonical_rows_by_id(split: str) -> dict[str, dict]:
        filename = EVAL_FILE_BY_SPLIT.get(split)
        path = data_root / filename if filename else None
        if path is None or not path.is_file():
            return {}
        return {row["id"]: row for row in rows(str(path))}

    def datasets():
        return [filename for filename in DATA_FILES if (data_root / filename).is_file()]

    def predictions():
        found: dict[str, list[str]] = {}
        if not output_root.is_dir():
            return found
        for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
            eval_dirs = [run_dir / "eval_final", run_dir / "eval_smoke", run_dir]
            splits = set()
            for eval_dir in eval_dirs:
                if eval_dir.is_dir():
                    splits.update(path.stem.removeprefix("predictions_") for path in eval_dir.glob("predictions_*.jsonl"))
            if splits:
                found[run_dir.name] = sorted(splits)
        return found

    def render(body: str, *, view: str, selected: str | None = None, selected_prediction=None):
        return render_template_string(
            PAGE, body=body, view=view, selected=selected, selected_prediction=selected_prediction,
            data_files=datasets(), prediction_index=predictions(),
        )

    def navigation(route: str, index: int, total: int):
        previous = url_for(route, **request.view_args, index=index - 1) if index else None
        following = url_for(route, **request.view_args, index=index + 1) if index + 1 < total else None
        return render_template_string(
            """<div class='card pager'><a class='button {% if not previous %}disabled{% endif %}' href='{{ previous or "#" }}'>← Previous</a><strong>Example {{ index + 1 }} of {{ total }}</strong><a class='button {% if not following %}disabled{% endif %}' href='{{ following or "#" }}'>Next →</a></div>""",
            previous=previous, following=following, index=index, total=total,
        )

    def example_card(row: dict | None, prediction: dict | None = None):
        row = row or {}
        components = prompt_components(row.get("prompt", ""))
        panels = "".join(
            f"<section class='panel {style}'><h3>{escape(title)}</h3><pre>{escape(text)}</pre></section>"
            for title, style, text in components
        )
        facts = "".join(f"<span class='fact'>{escape(key)}: {escape(str(value))}</span>" for key, value in (
            ("id", row.get("id")), ("split", row.get("split")), ("authorized", row.get("authorized")),
            ("candidate action", row.get("candidate_action")),
        ))
        prompt_card = "" if panels else "<div class='notice'>The matching canonical data row was not found, so the original prompt cannot be displayed.</div>"
        if panels:
            prompt_card = f"<div class='card'><h2 class='headline'>What the model was asked</h2><div class='panel-stack'>{panels}</div></div>"
        target = row.get("target", prediction.get("target") if prediction else None)
        ground_truth = f"<section class='panel truth json'><h3>Expected JSON action</h3><pre>{escape(json_block(target))}</pre></section>"
        if prediction is None:
            prediction_panel = ""
            raw_io = ""
        else:
            exact = bool(prediction.get("exact"))
            status = "exact target match" if exact else "does not match target"
            prediction_panel = f"""<section class='panel prediction json'><h3>Model-generated JSON <span class='status {'good' if exact else 'bad'}'>{status}</span></h3><pre>{escape(json_block(prediction.get('prediction')))}</pre></section>"""
            raw_prompt = row.get("prompt", "") + ("" if row.get("prompt", "").endswith("\n") else "\n")
            raw_io = f"""<details class='card raw-io'><summary>Show exact model input and generated completion</summary><p class='token-note'>Input is tokenized with <code>add_special_tokens=False</code>. The final newline below is the canonical separator. EOS stops generation and is not included in the decoded completion.</p><h3>Model input</h3><pre>{escape(raw_prompt)}</pre><h3>Decoded generated completion</h3><pre>{escape(str(prediction.get('raw_completion', '')))}</pre></details>"""
        metadata = row.get("metadata", prediction.get("metadata", {}) if prediction else {})
        return f"""<div class='facts'>{facts}</div>{prompt_card}<div class='panel-stack'>{ground_truth}{prediction_panel}</div>{raw_io}<div class='card json'><h3>Metadata</h3><pre>{escape(json_block(metadata))}</pre></div>"""

    @app.get("/")
    def home():
        body = """<div class='eyebrow'>Local, read-only viewer</div><h1 class='headline'>Inspect the experiment, one example at a time.</h1><p class='meta'>Browse canonical generated training/evaluation rows or compare synced model predictions with the exact JSON target.</p><div class='notice'>This app reads files only. It never loads a model, contacts a service, or changes reports.</div>"""
        return render(body, view="home")

    @app.get("/dataset/<path:filename>")
    def dataset(filename):
        if filename not in datasets():
            abort(404)
        records = rows(str(data_root / filename))
        index = page_index(request.args.get("index"), len(records))
        body = f"<div class='eyebrow'>Generated data</div><h1 class='headline'>{escape(filename)}</h1><p class='meta'>Canonical prompt and serialized target, before model inference.</p>"
        body += navigation("dataset", index, len(records)) + example_card(records[index])
        return render(body, view="dataset", selected=filename)

    @app.get("/prediction/<path:run>/<path:split>")
    def prediction(run, split):
        index_map = predictions()
        if split not in index_map.get(run, []):
            abort(404)
        run_dir = output_root / run
        candidates = [run_dir / "eval_final", run_dir / "eval_smoke", run_dir]
        path = next((directory / f"predictions_{split}.jsonl" for directory in candidates if (directory / f"predictions_{split}.jsonl").is_file()), None)
        if path is None:
            abort(404)
        records = rows(str(path))
        index = page_index(request.args.get("index"), len(records))
        record = records[index]
        canonical_row = canonical_rows_by_id(split).get(record.get("id"))
        body = f"<div class='eyebrow'>Synced model prediction</div><h1 class='headline'>{escape(run)}</h1><p class='meta'>Evaluation split: <strong>{escape(split)}</strong></p>"
        body += navigation("prediction", index, len(records)) + example_card(canonical_row, record)
        return render(body, view="prediction", selected_prediction=(run, split))

    return app


def main():
    parser = argparse.ArgumentParser(description="Browse generated data and synced evaluation predictions locally")
    parser.add_argument("--data-dir", default="authorization_dataset_v0/data/generated")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(args.data_dir, args.outputs_dir)
    print(f"Open http://{args.host}:{args.port} in a browser. Press Ctrl-C to stop.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
