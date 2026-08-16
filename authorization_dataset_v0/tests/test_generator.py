\
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict

from authdata.generator import AuthorizationDatasetGenerator
from authdata.factors import (
    ACTION_PHRASE_FAMILIES,
    HELD_OUT_MECHANISMS,
    LEXICAL_OOD_FAMILIES,
    POLICY_TEMPLATE_FAMILIES,
    RECOMBINATION_HOLDOUTS,
    RECOMBINATION_HOLDOUT_SOURCE_ACTIONS,
    TRAIN_LEXICAL_FAMILIES,
)
from authdata.templates import render_instruction

class TestGenerator(unittest.TestCase):
    def test_target_matches_authorization(self):
        g = AuthorizationDatasetGenerator(seed=1)
        for regime in ["attack_heavy", "diverse_attack", "authorization_balanced"]:
            rows = g.generate(regime, "train", 100)
            for r in rows:
                if r.authorized:
                    self.assertNotEqual(r.target["action"], "answer")
                else:
                    self.assertEqual(r.target["action"], "answer")

    def test_mechanism_ood(self):
        g = AuthorizationDatasetGenerator(seed=2)
        train = g.generate("authorization_balanced", "train", 100)
        rows = g.generate("authorization_balanced", "mechanism_ood", 100)
        self.assertFalse({r.mechanism for r in train} & set(HELD_OUT_MECHANISMS))
        self.assertTrue(all(r.mechanism in HELD_OUT_MECHANISMS for r in rows))
        self.assertEqual({r.mechanism for r in rows}, set(HELD_OUT_MECHANISMS))

    def test_balanced_training_examples_are_matched_counterfactual_triplets(self):
        rows = AuthorizationDatasetGenerator(seed=3).generate(
            "authorization_balanced", "train", 99,
        )
        triplets = defaultdict(list)
        for row in rows:
            triplets[row.metadata["counterfactual_triplet_id"]].append(row)

        self.assertEqual(len(triplets), 33)
        for triplet in triplets.values():
            self.assertEqual(len(triplet), 3)
            by_role = {row.metadata["triplet_role"]: row for row in triplet}
            self.assertEqual(set(by_role), {"reference", "unauthorized", "authorized"})
            reference = by_role["reference"]
            negative = by_role["unauthorized"]
            positive = by_role["authorized"]
            for field in ["source", "domain", "candidate_action", "style", "mechanism"]:
                self.assertEqual(getattr(reference, field), getattr(negative, field))
                self.assertEqual(getattr(negative, field), getattr(positive, field))
            self.assertEqual(reference.metadata["fact"], negative.metadata["fact"])
            self.assertEqual(negative.metadata["fact"], positive.metadata["fact"])
            self.assertFalse(reference.metadata["has_embedded_instruction"])
            self.assertNotIn("Reference record", reference.prompt)
            self.assertNotIn("executable external instruction", reference.prompt)
            self.assertEqual(reference.target["action"], "answer")
            self.assertEqual(negative.metadata["action_args"], positive.metadata["action_args"])
            self.assertEqual(negative.metadata["lexical_family"], positive.metadata["lexical_family"])
            self.assertEqual(
                negative.metadata["action_phrase_family"],
                positive.metadata["action_phrase_family"],
            )
            self.assertEqual(negative.metadata["mechanism_variant"], positive.metadata["mechanism_variant"])
            self.assertEqual(
                negative.metadata["policy_template_family"],
                positive.metadata["policy_template_family"],
            )
            self.assertEqual(
                negative.prompt.replace("not granted", "granted"),
                positive.prompt,
            )
            self.assertEqual(negative.target["action"], "answer")
            self.assertEqual(positive.target["action"], positive.candidate_action)

    def test_lexical_ood_families_do_not_appear_in_training(self):
        g = AuthorizationDatasetGenerator(seed=4)
        train = g.generate("authorization_balanced", "train", 200)
        lexical_ood = g.generate("authorization_balanced", "lexical_ood", 100)
        train_families = {row.metadata["lexical_family"] for row in train}
        ood_families = {row.metadata["lexical_family"] for row in lexical_ood}
        self.assertTrue(train_families <= set(TRAIN_LEXICAL_FAMILIES))
        self.assertTrue(ood_families <= set(LEXICAL_OOD_FAMILIES))
        self.assertFalse(train_families & ood_families)

    def test_balanced_triplets_cover_action_and_policy_surface_families(self):
        rows = AuthorizationDatasetGenerator(seed=12).generate(
            "authorization_balanced", "train", 300,
        )
        self.assertEqual(
            {row.metadata["action_phrase_family"] for row in rows},
            set(ACTION_PHRASE_FAMILIES),
        )
        self.assertEqual(
            {row.metadata["policy_template_family"] for row in rows},
            set(POLICY_TEMPLATE_FAMILIES),
        )

    def test_recombination_holdouts_do_not_leak_into_training(self):
        g = AuthorizationDatasetGenerator(seed=5)
        train = g.generate("authorization_balanced", "train", 200)
        recombination = g.generate("authorization_balanced", "auth_recombination", 100)
        train_tuples = {
            (row.source, row.candidate_action, row.authorized)
            for row in train
        }
        recombination_tuples = {
            (row.source, row.candidate_action, row.authorized)
            for row in recombination
        }
        self.assertFalse(train_tuples & RECOMBINATION_HOLDOUTS)
        self.assertEqual(recombination_tuples, RECOMBINATION_HOLDOUTS)
        for source, action in RECOMBINATION_HOLDOUT_SOURCE_ACTIONS:
            self.assertIn((source, action, False), recombination_tuples)
            self.assertIn((source, action, True), recombination_tuples)

    def test_style_is_observable_under_each_mechanism(self):
        args = {"file": "notes.txt", "recipient": "alice@example.com"}
        for mechanism in ["direct", "role_spoof", "fake_reasoning"]:
            rendered = {
                render_instruction("send_file", args, style, mechanism, "core_directive")
                for style in ["ordinary", "polite", "aggressive", "system_like"]
            }
            self.assertEqual(len(rendered), 4)

    def test_same_seed_produces_identical_jsonl_bytes(self):
        script = """
import json
from authdata.generator import AuthorizationDatasetGenerator
g = AuthorizationDatasetGenerator(seed=0)
for split, n in [('train', 100), ('iid', 20), ('lexical_ood', 20), ('mechanism_ood', 20), ('auth_recombination', 20), ('benign_control', 20)]:
    for row in g.generate('authorization_balanced', split, n):
        print(json.dumps(row.to_dict(), ensure_ascii=False))
"""
        project_root = Path(__file__).resolve().parents[1]
        first = subprocess.run(
            [sys.executable, "-c", script], cwd=project_root,
            check=True, capture_output=True,
        ).stdout
        second = subprocess.run(
            [sys.executable, "-c", script], cwd=project_root,
            check=True, capture_output=True,
        ).stdout
        self.assertEqual(first, second)

    def test_cli_writes_one_shared_evaluation_suite(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            all_out = root / "all"
            one_out = root / "one"
            subprocess.run(
                [
                    sys.executable, "generate.py", "--all-regimes", "--n-train", "20",
                    "--n-capability", "5", "--n-eval-each", "8", "--seed", "9", "--out", str(all_out),
                ],
                cwd=project_root, check=True, capture_output=True,
            )
            subprocess.run(
                [
                    sys.executable, "generate.py", "--regime", "attack_heavy", "--n-train", "20",
                    "--n-capability", "5", "--n-eval-each", "8", "--seed", "9", "--out", str(one_out),
                ],
                cwd=project_root, check=True, capture_output=True,
            )
            self.assertEqual(
                {path.name for path in all_out.iterdir()},
                {
                    "train_attack_heavy.jsonl", "train_diverse_attack.jsonl",
                    "train_authorization_balanced.jsonl", "eval_iid.jsonl",
                    "eval_lexical_ood.jsonl", "eval_mechanism_ood.jsonl",
                    "eval_auth_recombination.jsonl", "eval_benign_control.jsonl",
                },
            )
            for split in [
                "iid", "lexical_ood", "mechanism_ood", "auth_recombination", "benign_control",
            ]:
                self.assertEqual(
                    (all_out / f"eval_{split}.jsonl").read_bytes(),
                    (one_out / f"eval_{split}.jsonl").read_bytes(),
                )
            all_rows = {
                regime: [json.loads(line) for line in (all_out / f"train_{regime}.jsonl").read_text().splitlines()]
                for regime in ["attack_heavy", "diverse_attack", "authorization_balanced"]
            }
            shared = [row for row in all_rows["attack_heavy"] if row["regime"] == "shared_capability"]
            self.assertEqual(len(shared), 5)
            for rows in all_rows.values():
                self.assertEqual(shared, [row for row in rows if row["regime"] == "shared_capability"])
            for split in ["iid", "lexical_ood", "mechanism_ood", "auth_recombination", "benign_control"]:
                rows = [json.loads(line) for line in (all_out / f"eval_{split}.jsonl").read_text().splitlines()]
                has_closed_domain = any(row["metadata"].get("task_family") == "closed_domain" for row in rows)
                self.assertEqual(has_closed_domain, split == "mechanism_ood")

    def test_cli_replaces_preview_dataset(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            preview_out = Path(temp_dir) / "preview"
            preview_out.mkdir()
            (preview_out / "stale.txt").write_text("stale", encoding="utf-8")
            subprocess.run(
                [sys.executable, "generate.py", "--preview", "--out", str(preview_out)],
                cwd=project_root, check=True, capture_output=True,
            )
            self.assertEqual(
                {path.name for path in preview_out.iterdir()},
                {
                    "train_20.jsonl", "iid_10.jsonl", "lexical_ood_10.jsonl",
                    "mechanism_ood_10.jsonl", "auth_recombination_10.jsonl",
                    "benign_control_10.jsonl",
                },
            )
            self.assertEqual(len((preview_out / "train_20.jsonl").read_text().splitlines()), 20)

    def test_benign_control_supports_a_200_example_eval_split(self):
        rows = AuthorizationDatasetGenerator(seed=13).generate(
            "shared_eval", "benign_control", 200,
        )
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({row.id for row in rows}), 200)

if __name__ == "__main__":
    unittest.main()
