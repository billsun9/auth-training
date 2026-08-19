from results_browser import page_index, prompt_components


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
