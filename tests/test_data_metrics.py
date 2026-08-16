import json
from pathlib import Path
from auth_sft.data import canonical_target, prompt_with_separator, read_jsonl
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
