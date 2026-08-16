import json
from pathlib import Path
from auth_sft.data import canonical_target, prompt_with_separator, read_jsonl, split_train_validation_rows
from auth_sft.metrics import compute_metrics, extract_json_object

U={"id":"u","split":"auth_recombination","regime":"shared_eval","source":"email",
"authorized":False,"style":"ordinary","mechanism":"role_spoof","domain":"web",
"candidate_action":"send_file","prompt":"test","target":{"action":"answer","content":"fact"},
"metadata":{"counterfactual_pair_id":"pair","action_args":{"file":"x","recipient":"a@b.com"}}}
A={**U,"id":"a","authorized":True,
"target":{"action":"send_file","file":"x","recipient":"a@b.com"}}

def test_extract_json():
    assert extract_json_object('x {"action":"answer","content":"y"} z') == {"action":"answer","content":"y"}

def test_pair_metrics():
    rows=[U,A]; preds=[r["target"] for r in rows]
    m=compute_metrics(rows,[canonical_target(p) for p in preds],preds)
    assert m["exact_target_accuracy"]==1.0
    assert m["unauthorized_execution_rate"]==0.0
    assert m["counterfactual_pairs"]["pair_exact_accuracy"]==1.0


def test_triplet_metrics_and_pair_compatibility():
    triplet_id = "triplet"
    reference = {**U, "id": "r", "metadata": {"counterfactual_pair_id": triplet_id,
        "counterfactual_triplet_id": triplet_id, "triplet_role": "reference"}}
    unauthorized = {**U, "id": "u", "metadata": {"counterfactual_pair_id": triplet_id,
        "counterfactual_triplet_id": triplet_id, "triplet_role": "unauthorized"}}
    authorized = {**A, "id": "a", "metadata": {"counterfactual_pair_id": triplet_id,
        "counterfactual_triplet_id": triplet_id, "triplet_role": "authorized"}}
    rows = [reference, unauthorized, authorized]
    preds = [row["target"] for row in rows]
    metrics = compute_metrics(rows, [canonical_target(pred) for pred in preds], preds)
    assert metrics["counterfactual_pairs"]["n_pairs"] == 1
    assert metrics["counterfactual_triplets"]["n_triplets"] == 1
    assert metrics["counterfactual_triplets"]["triplet_exact_accuracy"] == 1.0

def test_unauthorized_execution():
    p={"action":"send_file","file":"x","recipient":"a@b.com"}
    assert compute_metrics([U],[canonical_target(p)],[p])["unauthorized_execution_rate"]==1.0

def test_benign_answer_is_not_an_unauthorized_execution():
    row = {**U, "candidate_action":"answer"}
    pred = {"action":"answer", "content":"fact"}
    metrics = compute_metrics([row], [canonical_target(pred)], [pred])
    assert metrics["unauthorized_execution_n"] == 0
    assert metrics["unauthorized_execution_rate"] is None

def test_read_schema(tmp_path:Path):
    p=tmp_path/"x.jsonl"; p.write_text(json.dumps(U)+"\n",encoding="utf-8")
    assert read_jsonl(p)[0]["id"]=="u"


def test_validation_split_is_deterministic_and_keeps_pairs_together():
    rows = [
        {**U, "id": "u1", "metadata": {"counterfactual_pair_id": "pair-1"}},
        {**A, "id": "a1", "metadata": {"counterfactual_pair_id": "pair-1"}},
        {**U, "id": "u2", "metadata": {"counterfactual_pair_id": "pair-2"}},
        {**A, "id": "a2", "metadata": {"counterfactual_pair_id": "pair-2"}},
    ]
    train, validation = split_train_validation_rows(rows, validation_ratio=0.25, seed=7)
    again_train, again_validation = split_train_validation_rows(rows, validation_ratio=0.25, seed=7)
    assert {row["id"] for row in train} == {row["id"] for row in again_train}
    assert {row["id"] for row in validation} == {row["id"] for row in again_validation}
    train_pairs = {row["metadata"]["counterfactual_pair_id"] for row in train}
    validation_pairs = {row["metadata"]["counterfactual_pair_id"] for row in validation}
    assert not train_pairs & validation_pairs


def test_validation_split_keeps_triplets_together():
    rows = []
    for triplet_id in ("triplet-1", "triplet-2"):
        for role, row in (("reference", U), ("unauthorized", U), ("authorized", A)):
            rows.append({**row, "id": f"{triplet_id}-{role}", "metadata": {
                "counterfactual_triplet_id": triplet_id, "triplet_role": role,
            }})
    train, validation = split_train_validation_rows(rows, validation_ratio=0.5, seed=1)
    train_ids = {row["metadata"]["counterfactual_triplet_id"] for row in train}
    validation_ids = {row["metadata"]["counterfactual_triplet_id"] for row in validation}
    assert not train_ids & validation_ids
    assert len(train) == len(validation) == 3


def test_completion_only_mask():
    from auth_sft.data import PromptCompletionDataset

    class FakeTokenizer:
        eos_token_id = 99
        eos_token = "<eos>"
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False, truncation=False):
            # deterministic one-char -> one-token toy tokenizer
            return {"input_ids": [ord(c) % 97 + 1 for c in text]}

    row = {
        "id":"x","split":"train","regime":"authorization_balanced",
        "source":"email","authorized":False,"style":"ordinary","mechanism":"direct",
        "domain":"web","candidate_action":"send_file","prompt":"PROMPT",
        "target":{"action":"answer","content":"fact"},"metadata":{}
    }
    tok = FakeTokenizer()
    ds = PromptCompletionDataset([row], tok, max_length=512)
    ex = ds[0]
    prompt_len = len(tok("PROMPT\n")["input_ids"])
    assert ex["labels"][:prompt_len].tolist() == [-100] * prompt_len
    assert ex["labels"][prompt_len:].tolist() == ex["input_ids"][prompt_len:].tolist()
    assert ex["labels"][-1].item() == 99

def test_prompt_serialization_preserves_rendered_text():
    assert prompt_with_separator("PROMPT  ") == "PROMPT  \n"
    assert prompt_with_separator("PROMPT\n") == "PROMPT\n"

def test_overflow_is_rejected_without_truncation():
    class TinyTokenizer:
        eos_token_id = 99
        eos_token = "<eos>"
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False, truncation=False):
            return {"input_ids": list(range(1, len(text) + 1))}
    row = {**U, "split":"train", "regime":"authorization_balanced"}
    from auth_sft.data import PromptCompletionDataset
    try:
        PromptCompletionDataset([row], TinyTokenizer(), max_length=2)
    except ValueError as e:
        assert "truncation" in str(e).lower()
    else:
        raise AssertionError("overflow must not be silently truncated")
