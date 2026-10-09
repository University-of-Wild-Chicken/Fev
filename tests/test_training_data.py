"""Synthetic labels, split isolation and frozen training-input regressions."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from fev.ir import content_hash
from scripts.generate_training_data import (FAMILIES, all_spec_outputs, family_case,
    generate, make_records, oracle, parameters, sha_file, unsupported_records,
    validate_record, verify_parent)
from scripts.train_lora import make_input_loader, select_rows


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.directory.name) / "dataset"
        cls.manifest = generate(cls.dataset, parents_per_family=2, surfaces=2)
        cls.rows = {split: [json.loads(line) for line in (cls.dataset / (split + ".jsonl")).read_text().splitlines()]
                    for split in ("train", "validation", "test")}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_every_supported_family_has_independently_checked_material_witness(self):
        for family in FAMILIES:
            if family.startswith("unsupported_"):
                continue
            with self.subTest(family=family):
                p, evidence = verify_parent(family, 1)
                witness = evidence["canonical_witness"]
                self.assertNotEqual(oracle(family, 0, witness["input"], p),
                                    oracle(family, 1, witness["input"], p))
                self.assertEqual(evidence["material"]["result"], "distinct")
                self.assertEqual(evidence["duplicate"]["result"], "equivalent_in_scope")
                self.assertTrue(evidence["material"]["witness_replayed"])

    def test_boundaries_and_short_circuit_error_semantics(self):
        p = parameters(0)
        self.assertEqual(oracle("rank_duplicate_policy", 0, [2, 2, 1], p), {"kind": "value", "value": 2})
        self.assertEqual(oracle("rank_duplicate_policy", 1, [2, 2, 1], p), {"kind": "value", "value": 1})
        self.assertEqual(oracle("compound_predicate", 0, [], p), {"kind": "value", "value": False})
        self.assertEqual(oracle("compound_predicate", 1, [], p), {"kind": "error", "code": "index_out_of_bounds"})

    def test_source_resolved_variant_keeps_only_the_explicit_choice(self):
        p, evidence = verify_parent("rank_duplicate_policy", 1)
        rows = make_records("rank_duplicate_policy", 1, 3, p, evidence)
        proposal = next(row for row in rows if row["stage"] == "proposal")
        target = json.loads(proposal["messages"][-1]["content"])
        self.assertEqual(len(target["forks"]), 1)
        self.assertEqual(target["forks"][0]["selected_axis_options"], {"rank_duplicate_policy": "option_1"})
        self.assertTrue(any(assumption["provenance"] == "explicit_text" for assumption in target["forks"][0]["assumptions"]))

    def test_malformed_source_spans_and_unsupported_provenance_rejected(self):
        row = deepcopy(next(row for row in self.rows["train"] if row["stage"] == "intake"))
        target = json.loads(row["messages"][-1]["content"])
        target["claims"][0]["end"] -= 1
        row["messages"][-1]["content"] = json.dumps(target)
        with self.assertRaisesRegex(ValueError, "span mismatch"):
            validate_record(row)
        target["claims"][0]["provenance"] = "user_confirmed"
        row["messages"][-1]["content"] = json.dumps(target)
        with self.assertRaisesRegex(ValueError, "enum mismatch"):
            validate_record(row)

    def test_unsupported_targets_preserve_requirement_without_fabricated_spec(self):
        for family in FAMILIES:
            if not family.startswith("unsupported_"):
                continue
            for row in unsupported_records(family, count=1):
                target = json.loads(row["messages"][-1]["content"])
                self.assertTrue(target["unsupported_requirements"])
                self.assertEqual(validate_record(row), [])
                self.assertFalse(target.get("forks"))

    def test_splits_separate_parent_family_and_output_ast(self):
        for left, right in itertools.combinations(self.rows, 2):
            for field in ("family", "parent_id"):
                self.assertFalse({r[field] for r in self.rows[left]} & {r[field] for r in self.rows[right]})
            left_outputs = set().union(*(all_spec_outputs(row) for row in self.rows[left]))
            right_outputs = set().union(*(all_spec_outputs(row) for row in self.rows[right]))
            self.assertFalse(left_outputs & right_outputs)

    def test_manifest_hashes_and_message_deduplication(self):
        for name, expected in self.manifest["files_sha256"].items():
            self.assertEqual(sha_file(self.dataset / name), expected)
        messages = [content_hash(row["messages"]) for rows in self.rows.values() for row in rows]
        self.assertEqual(len(messages), len(set(messages)))
        with self.assertRaisesRegex(ValueError, "not empty"):
            generate(self.dataset)

    def test_production_repair_packet_and_concrete_frontier(self):
        for rows in self.rows.values():
            for row in rows:
                if row["stage"] != "repair":
                    continue
                packet = json.loads(row["messages"][1]["content"])["packet"]
                self.assertEqual(packet["compiler_diagnostic"]["kind"], "compiler_diagnostic")
                self.assertIn("candidate", packet["compiler_diagnostic"])
                target = json.loads(row["messages"][-1]["content"])
                self.assertEqual(target["search_frontier"], [])

    def test_pilot_selection_balances_stages_and_never_uses_test(self):
        rows = select_rows(self.rows["train"], 20, 123)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r["id"] for r in rows}), 20)
        self.assertEqual({r["split"] for r in rows}, {"train"})
        self.assertEqual({stage: sum(r["stage"] == stage for r in rows)
                          for stage in ("intake", "proposal", "repair", "critic")},
                         {"intake": 5, "proposal": 5, "repair": 5, "critic": 5})

    def test_pilot_loader_is_byte_bound_and_refuses_test(self):
        prepared = Path(self.directory.name) / "pilot"
        prepared.mkdir(exist_ok=True)
        captured = (json.dumps(self.rows["train"][0]) + "\n").encode()
        index_bytes = b'{"id":"lineage"}\n'
        (prepared / "train.jsonl").write_bytes(captured)
        (prepared / "train-index.jsonl").write_bytes(index_bytes)
        config = {"files_sha256": {"train.jsonl": hashlib.sha256(captured).hexdigest(),
                                    "train-index.jsonl": hashlib.sha256(index_bytes).hexdigest()},
                  "selected_ids": {"train": [self.rows["train"][0]["id"]]},
                  "source_manifest_sha256": "fixture-manifest"}
        loader = make_input_loader(prepared.resolve(), config)
        loaded = loader(prepared / "train.jsonl")
        self.assertEqual(loaded["lineage"]["dataset_sha256"], hashlib.sha256(captured).hexdigest())
        with self.assertRaisesRegex(ValueError, "only frozen"):
            loader(prepared / "test.jsonl")
        (prepared / "train.jsonl").write_bytes(captured + b"\n")
        with self.assertRaisesRegex(ValueError, "data changed"):
            loader(prepared / "train.jsonl")


if __name__ == "__main__":
    unittest.main()
