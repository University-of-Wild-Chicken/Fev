from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import sqlite3

import pytest

from fev.commitments import CommitmentError, IntegrityError, Ledger, StaleCommitmentError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def run(request_id="request-1"):
    specification = {"version": 1, "input": {"kind": "list", "items": {"kind": "int", "min": -8, "max": 8},
                                            "min_length": 0, "max_length": 5},
                     "output": {"op": "at", "values": {"op": "input"}, "index": 1, "error": "too_few_values"}}
    result = {"request_id": request_id,
            "task": {"scope": {"min_value": -8, "max_value": 8, "min_length": 0, "max_length": 5},
                     "observation_schema": {"version": 1, "effects": "pure"}, "environment": {"predicates": []}},
            "forks": [{"fork_id": "F1", "status": "valid", "spec_hash": digest(specification),
                       "typed_spec_ir": specification,
                       "assumptions": [{"text": "duplicates retained", "provenance": "inferred"}],
                       "unsupported_requirements": [], "unresolved_axes": [],
                       "consistency_evidence": {"result": "sat"}}],
            "search_frontier": [], "unresolved_candidates": [], "limitations": ["Bounded domain only"],
            "decision": "clarify"}
    result["forks"][0]["consistency_evidence"].update(
        spec_hashes=[digest(specification)], scope_hash=digest(result["task"]["scope"]),
        environment_hash=digest(result["task"]["environment"]),
        observation_hash=digest(result["task"]["observation_schema"]),
        status="consistent_in_scope", witness_replayed=True,
        scope=copy.deepcopy(result["task"]["scope"]), environment=copy.deepcopy(result["task"]["environment"]),
        observation_schema=copy.deepcopy(result["task"]["observation_schema"]),
        runtime_manifest={"kernel": "test-fixture"}, runtime_hash=digest({"kernel": "test-fixture"}))
    return result


def save(ledger, result):
    for fork in result.get("forks", []):
        fork["consistency_evidence_ref"] = ledger.put_artifact(fork["consistency_evidence"])
    return ledger.save_run(result)


def select(ledger, **kwargs):
    return ledger.commit("request-1", "F1", "user_decision", "I select F1 and its documented boundary behavior.", **kwargs)


def test_roundtrip_frozen_versions_and_exact_context(tmp_path):
    ledger = Ledger(tmp_path)
    original = run()
    run_hash = save(ledger, original)
    selected = select(ledger, expected_run_hash=run_hash)
    assert selected["run_hash"] == run_hash
    assert selected["selection_provenance"]["provenance"] == "user_confirmed"
    assert selected["assumptions"][0]["provenance"] == "inferred"
    assert [error["code"] for error in selected["error_cases"]] == ["invalid_input", "input_out_of_bounds", "too_few_values"]
    for key in ("scope", "environment", "observation_schema"):
        assert selected[key] == original["task"][key]
        assert ledger.get_artifact(selected[f"{key}_hash"]) == selected[key]
    original["forks"][0]["typed_spec_ir"]["output"]["index"] = 0
    selected["typed_spec_ir"]["output"]["index"] = 99
    assert Ledger(tmp_path).get_commitment("request-1")["typed_spec_ir"]["output"]["index"] == 1
    assert ledger.get_run("request-1")["run_revision"] == 1
    assert ledger.run_hash("request-1") == run_hash


def test_changed_contract_invalidates_old_evidence_without_deleting_it(tmp_path):
    ledger = Ledger(tmp_path)
    save(ledger, run())
    first = select(ledger)
    evidence = ledger.save_evidence(first["commitment_id"], {"kind": "runtime", "result": "pass"})
    second_run = run()
    second_run["forks"][0]["typed_spec_ir"]["output"]["index"] = 0
    second_run["forks"][0]["spec_hash"] = digest(second_run["forks"][0]["typed_spec_ir"])
    second_run["forks"][0]["consistency_evidence"]["spec_hashes"] = [second_run["forks"][0]["spec_hash"]]
    save(ledger, second_run)
    second = select(ledger)
    assert second["version"] == 2
    assert ledger.get_commitment(first["commitment_id"])["version"] == 1
    old = ledger.list_evidence(first["commitment_id"])[0]
    assert old["evidence_id"] == evidence["evidence_id"]
    assert old["data"] == {"kind": "runtime", "result": "pass"}
    assert old["valid"] is False
    assert old["invalidated_reason"] == "commitment_superseded"
    with pytest.raises(StaleCommitmentError):
        ledger.assert_current(first["commitment_id"])
    with pytest.raises(StaleCommitmentError):
        ledger.save_evidence(first["commitment_id"], {"result": "pass"})
    assert ledger.assert_current(second["commitment_id"])["version"] == 2


def test_proposal_revision_cannot_rewrite_committed_contract(tmp_path):
    ledger = Ledger(tmp_path)
    first_hash = save(ledger, run())
    selected = select(ledger)
    newer = run()
    newer["limitations"].append("new proposed limitation")
    save(ledger, newer)
    with pytest.raises(StaleCommitmentError):
        select(ledger, expected_run_hash=first_hash)
    assert ledger.get_commitment(selected["commitment_id"])["run_hash"] == first_hash
    assert ledger.assert_current(selected["commitment_id"])


@pytest.mark.parametrize("change", [
    {"status": "duplicate"}, {"status": "invalid"}, {"status": "unresolved"},
    {"unsupported_requirements": ["arbitrary precision"]}, {"unresolved_axes": ["error behavior"]},
    {"consistency_evidence": {"result": "unsat"}}, {"consistency_evidence": {"result": "unknown"}},
])
def test_rejects_unresolved_or_ineligible_contracts(tmp_path, change):
    ledger = Ledger(tmp_path)
    result = run()
    result["forks"][0].update(change)
    save(ledger, result)
    with pytest.raises(CommitmentError):
        select(ledger, accept_limitations=True)


def test_duplicate_fork_identity_and_changed_spec_hash_are_rejected(tmp_path):
    ledger = Ledger(tmp_path)
    result = run()
    result["forks"].append(copy.deepcopy(result["forks"][0]))
    save(ledger, result)
    with pytest.raises(CommitmentError, match="exactly once"):
        select(ledger)
    result = run()
    result["forks"][0]["typed_spec_ir"]["output"]["index"] = 2
    save(ledger, result)
    with pytest.raises(IntegrityError, match="specification hash"):
        select(ledger)


@pytest.mark.parametrize("field,value", [
    ("search_frontier", [{"reason": "budget_exhausted"}]),
    ("unresolved_candidates", [{"fork_id": "F2"}]),
    ("unsupported_requirements", ["network effects"]),
    ("comparisons", [{"result": "unresolved"}]),
])
def test_frontier_requires_explicit_user_acknowledgment(tmp_path, field, value):
    ledger = Ledger(tmp_path)
    result = run()
    result[field] = value
    save(ledger, result)
    with pytest.raises(CommitmentError, match="acknowledgment"):
        select(ledger)
    committed = select(ledger, accept_limitations=True)
    assert committed["selection_provenance"]["accepted_limitations"] is True


def test_model_or_policy_cannot_manufacture_user_confirmation(tmp_path):
    ledger = Ledger(tmp_path)
    save(ledger, run())
    for source in ("model", "user_confirmed", "explicit_policy"):
        with pytest.raises(CommitmentError, match="actual user_decision"):
            ledger.commit("request-1", "F1", source, "F1")
    forged = run()
    forged["forks"][0]["assumptions"][0]["provenance"] = "user_confirmed"
    save(ledger, forged)
    with pytest.raises(CommitmentError, match="prior commitment_id"):
        select(ledger)


def test_unrelated_prior_selection_cannot_confirm_an_assumption(tmp_path):
    ledger = Ledger(tmp_path)
    save(ledger, run())
    first = select(ledger)
    forged = run()
    forged["forks"][0]["assumptions"] = [{"text": "silently discard failures", "provenance": "user_confirmed",
                                          "commitment_id": first["commitment_id"]}]
    save(ledger, forged)
    with pytest.raises(CommitmentError, match="does not confirm"):
        select(ledger)


def test_artifact_tampering_and_unsafe_reference_are_detected(tmp_path):
    ledger = Ledger(tmp_path)
    hashed = ledger.put_artifact({"α": [1, 2]})
    assert ledger.put_artifact({"α": [1, 2]}) == hashed
    (ledger.artifacts_dir / f"{hashed}.json").write_text('{"tampered":true}')
    with pytest.raises(IntegrityError):
        ledger.get_artifact(hashed)
    with pytest.raises(IntegrityError):
        ledger.put_artifact({"α": [1, 2]})
    with pytest.raises(IntegrityError):
        ledger.get_artifact("../../secret")


def test_referenced_evidence_must_exist_and_remain_intact(tmp_path):
    ledger = Ledger(tmp_path)
    result = run()
    result["forks"][0]["consistency_evidence"]["evidence_ref"] = "a" * 64
    with pytest.raises(IntegrityError):
        save(ledger, result)
    ref = ledger.put_artifact({"result": "sat"})
    result["forks"][0]["consistency_evidence"]["evidence_ref"] = ref
    save(ledger, result)
    selected = select(ledger)
    (ledger.artifacts_dir / f"{ref}.json").unlink()
    with pytest.raises(IntegrityError):
        ledger.get_commitment(selected["commitment_id"])


def test_evidence_bound_to_different_scope_is_rejected(tmp_path):
    ledger = Ledger(tmp_path)
    result = run()
    result["forks"][0]["consistency_evidence"]["scope_hash"] = digest({"different": True})
    save(ledger, result)
    with pytest.raises(IntegrityError, match="scope_hash"):
        select(ledger)


@pytest.mark.parametrize("field,value", [
    ("spec_hashes", ["f" * 64]),
    ("observation_hash", "f" * 64),
    ("environment_hash", "f" * 64),
    ("runtime_hash", "f" * 64),
    ("scope", {"forged": "scope"}),
])
def test_other_consistency_bindings_cannot_be_substituted(tmp_path, field, value):
    ledger = Ledger(tmp_path)
    result = run()
    result["forks"][0]["consistency_evidence"][field] = value
    save(ledger, result)
    with pytest.raises(IntegrityError):
        select(ledger)


def test_inline_consistency_must_equal_referenced_artifact(tmp_path):
    ledger = Ledger(tmp_path)
    result = run()
    save(ledger, result)
    result["forks"][0]["consistency_evidence"]["elapsed_ms"] = 42
    ledger.save_run(result)
    with pytest.raises(IntegrityError, match="stored artifact"):
        select(ledger)


def test_missing_consistency_reference_is_rejected(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.save_run(run())
    with pytest.raises(IntegrityError, match="stored artifact"):
        select(ledger)


def test_immutable_sql_rows_and_foreign_keys(tmp_path):
    ledger = Ledger(tmp_path)
    save(ledger, run())
    selected = select(ledger)
    ledger.save_evidence(selected["commitment_id"], {"result": "pass"})
    with sqlite3.connect(ledger.database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        for table in ("artifacts", "runs", "commitments", "evidence"):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute(f"DELETE FROM {table}")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("INSERT INTO evidence VALUES ('fake', 'missing', 'missing')")


def test_concurrent_commits_have_unique_serial_versions(tmp_path):
    ledger = Ledger(tmp_path)
    expected_hash = save(ledger, run())
    with ThreadPoolExecutor(max_workers=4) as executor:
        selected = list(executor.map(lambda _: select(ledger, expected_run_hash=expected_hash), range(8)))
    assert sorted(item["version"] for item in selected) == list(range(1, 9))
    assert len({item["commitment_id"] for item in selected}) == 8
    assert ledger.get_commitment("request-1")["version"] == 8
