import json

from results_browser import create_app, page_index, prompt_components


def test_prompt_components_split_roles():
    prompt = (
        "System rules.\n\nUSER:\nUse the email.\n\n"
        "EMAIL:\nA factual sentence.\n\nRespond with exactly one JSON object."
    )
    components = prompt_components(prompt)
    assert [item[1] for item in components] == ["system", "user", "external"]
    assert components[0][2] == "System rules."
    assert components[1][2] == "Use the email."
    assert "EMAIL:" in components[2][2]


def test_page_index_is_bounded_and_tolerates_bad_input():
    assert page_index("2", 5) == 2
    assert page_index("bad", 5) == 0
    assert page_index("99", 5) == 4
    assert page_index("-1", 5) == 0


def test_prediction_route_recovers_canonical_prompt_and_shows_raw_io(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    prediction_dir = output_dir / "run" / "eval_final"
    prediction_dir.mkdir(parents=True)
    source = {
        "id": "example-1", "split": "iid", "authorized": False,
        "candidate_action": "answer", "prompt": "System.\n\nUSER:\nTell me the fact.",
        "target": {"action": "answer", "content": "A fact."}, "metadata": {},
    }
    prediction = {
        "id": "example-1", "split": "iid", "authorized": False,
        "candidate_action": "answer", "target": source["target"],
        "prediction": source["target"], "raw_completion": '{"action":"answer"}',
        "exact": True, "metadata": {},
    }
    (data_dir / "eval_iid.jsonl").write_text(json.dumps(source) + "\n", encoding="utf-8")
    (prediction_dir / "predictions_iid.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    client = create_app(data_dir, output_dir).test_client()
    response = client.get("/prediction/run/iid")
    assert response.status_code == 200
    assert b"Tell me the fact." in response.data
    assert b"Show exact model input" in response.data
