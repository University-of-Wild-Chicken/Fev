from copy import deepcopy
import json

import pytest

from fev.commitments import Ledger, CommitmentError, StaleCommitmentError
from fev.fork_search import ForkService
from fev.implementation import execute
from fev.intake import validate_claim
from fev.model_adapter import ReplayProposer
from fev.schemas import Budgets


def selector(distinct):
    values = {"op": "input"}
    if distinct:
        values = {"op": "unique", "values": values}
    return {"label": "distinct" if distinct else "multiplicity", "assumptions": [
        {"text": "Too few values return index_out_of_bounds; input is a bounded integer list.", "provenance": "external_assumption"}],
        "selected_axis_options": {"duplicates": "distinct" if distinct else "multiplicity"},
        "typed_spec_ir": {"version": 1, "input": {"kind": "list", "items": {"kind": "int", "min": -8, "max": 8},
        "min_length": 0, "max_length": 5}, "output": {"op": "at", "index": 1,
        "values": {"op": "sort", "order": "desc", "values": values}}}}


def records(candidates=None):
    return [{"stage": "intake", "data": {"claims": [{"text": "Return the second-largest number.", "start": 0, "end": 33,
                "formal_meaning": "Select rank two", "provenance": "explicit_text"}]}},
            {"stage": "proposal", "data": {"forks": candidates or [selector(True)]}},
            {"stage": "proposal", "data": {"forks": [selector(False)]}},
            {"stage": "expansion", "data": {"forks": []}},
            {"stage": "expansion", "data": {"forks": []}},
            {"stage": "critic", "data": {"critic_notes": ["Bounded domain; no completeness claim."]}}]


def analyze(tmp_path, data=None, budgets=None):
    ledger = Ledger(tmp_path)
    result = ForkService(ReplayProposer(data or records()), ledger, budgets).analyze("Return the second-largest number.")
    return ledger, result


def test_full_flow_distinguishes_commits_and_invalidates(tmp_path):
    ledger, result = analyze(tmp_path)
    assert result["decision"] == "clarify"
    assert [f["status"] for f in result["forks"]] == ["valid", "valid"]
    assert result["comparisons"][0]["result"] == "distinct"
    assert result["scores"]["outside_retained_set_probability"] is None
    assert len([c for c in result["action_matrix"] if c["conformance"] == "violated"]) == 2
    committed = ledger.commit(result["request_id"], "F1", "user_decision", "Test user selects distinct", expected_run_hash=result["run_hash"])
    assert execute(ledger, committed["commitment_id"], [5, 5, 3])["result"] == {"kind": "value", "value": 3}
    assert execute(ledger, committed["commitment_id"], [5, 5])["result"]["kind"] == "error"
    replacement = ledger.commit(result["request_id"], "F2", "user_decision", "Test user changes to multiplicity")
    assert execute(ledger, replacement["commitment_id"], [5, 5, 3])["result"]["value"] == 5
    with pytest.raises(StaleCommitmentError):
        execute(ledger, committed["commitment_id"], [5, 5, 3])


def test_cap_retains_last_completed_response(tmp_path):
    _, result = analyze(tmp_path, budgets=Budgets(max_model_calls=2))
    assert result["resource_usage"]["model_calls"] == 2
    assert len(result["forks"]) == 1
    assert result["stop_reason"] == "model_call_budget"
    assert result["search_frontier"]


def test_duplicates_merge_only_with_comparison(tmp_path):
    data = records()
    data[2]["data"]["forks"] = [selector(True)]
    _, result = analyze(tmp_path, data)
    assert result["forks"][1]["status"] == "duplicate"
    assert result["comparisons"][0]["result"] == "equivalent_in_scope"
    assert result["decision"] == "defer"


def test_unsupported_fork_cannot_commit_even_with_acknowledgment(tmp_path):
    candidate = selector(True)
    candidate["unsupported_requirements"] = ["floating point NaN"]
    ledger, result = analyze(tmp_path, records([candidate]))
    assert result["forks"][0]["status"] == "unresolved"
    with pytest.raises(CommitmentError):
        ledger.commit(result["request_id"], "F1", "user_decision", "select", accept_limitations=True)


def test_model_cannot_confirm_user_intent(tmp_path):
    candidate = selector(True)
    candidate["assumptions"][0]["provenance"] = "user_confirmed"
    data = records([candidate]) + [{"stage": "repair", "data": {"forks": []}}]
    _, result = analyze(tmp_path, data)
    assert result["unresolved_candidates"]
    assert all(assumption["provenance"] != "user_confirmed"
               for fork in result["forks"] for assumption in fork["assumptions"])
    assert all(fork["label"] != "distinct" for fork in result["forks"])


def test_source_span_is_not_accepted_by_paraphrase():
    with pytest.raises(ValueError, match="does not match"):
        validate_claim("return x", {"text": "return y", "start": 0, "end": 8, "provenance": "explicit_text"})


def test_zero_solver_budget_never_creates_valid_contract(tmp_path):
    _, result = analyze(tmp_path, budgets=Budgets(solver_total_ms=0))
    assert not result["forks"]
    assert result["decision"] == "defer"
    assert result["stop_reason"] == "solver_budget"


def test_exported_replay_is_usable(tmp_path):
    ledger, original = analyze(tmp_path / "first")
    saved = [ledger.get_artifact(ref) for ref in original["evidence_refs"]]
    _, replay = analyze(tmp_path / "second", saved)
    assert replay["decision"] == original["decision"]
    assert [f["spec_hash"] for f in replay["forks"]] == [f["spec_hash"] for f in original["forks"]]


def test_changed_interpreter_cannot_execute_prior_commitment(tmp_path, monkeypatch):
    from fev import implementation
    ledger, result = analyze(tmp_path)
    commitment = ledger.commit(result["request_id"], "F1", "user_decision", "test selection")
    runtime = implementation.runtime_manifest()
    runtime["source_hashes"]["interpreter.py"] = "changed"
    monkeypatch.setattr(implementation, "runtime_manifest", lambda: runtime)
    with pytest.raises(ValueError, match="runtime changed"):
        execute(ledger, commitment["commitment_id"], [5, 5, 3])


def test_unavailable_model_still_saves_explicit_deferred_result(tmp_path):
    class Unavailable:
        def propose(self, *args, **kwargs):
            raise RuntimeError("endpoint unavailable")

        def manifest(self):
            raise RuntimeError("endpoint unavailable")

    ledger = Ledger(tmp_path)
    result = ForkService(Unavailable(), ledger).analyze("Return the largest number.")
    assert result["decision"] == "defer"
    assert result["task"]["model_manifest"]["status"] == "unavailable"
    assert ledger.get_run(result["request_id"])["unresolved_candidates"]


def test_critic_cannot_inject_new_forks_when_runtime_ignores_schema(tmp_path):
    data = records()
    data[-1]["data"] = {"forks": [selector(True)]}
    _, result = analyze(tmp_path, data)
    assert len(result["forks"]) == 2
    assert any(c["kind"] == "response_diagnostic" and c["stage"] == "critic" for c in result["unresolved_candidates"])


def test_unresolved_consistency_remains_in_frontier(tmp_path, monkeypatch):
    import fev.fork_search as search
    actual = search.check_consistency
    count = 0
    def interrupted(spec, scope, timeout_ms):
        nonlocal count
        count += 1
        if count == 2:
            return {"result": "unknown", "reason": "timeout"}
        return actual(spec, scope, timeout_ms)
    monkeypatch.setattr(search, "check_consistency", interrupted)
    ledger, result = analyze(tmp_path)
    assert [f["status"] for f in result["forks"]] == ["valid", "unresolved"]
    assert any(item["kind"] == "unresolved_consistency" for item in result["search_frontier"])
    with pytest.raises(CommitmentError):
        ledger.commit(result["request_id"], "F1", "user_decision", "Select first")


@pytest.mark.parametrize("extra", [{"unsupported_requirements": "floats"}, {"unmodeled_requirements": ["floats"]}])
def test_malformed_or_unknown_obligations_cannot_disappear(tmp_path, extra):
    data = records()
    data[1]["data"].update(extra)
    _, result = analyze(tmp_path, data)
    assert result["unresolved_candidates"]
    assert result["search_frontier"]
    assert len(result["forks"]) == 1


def test_execution_respects_frozen_scope_and_binds_evidence(tmp_path):
    from fev.ir import Scope
    ledger = Ledger(tmp_path)
    result = ForkService(ReplayProposer(records()), ledger).analyze(
        "Return the second-largest number.", scope=Scope(min_value=-4, max_value=4))
    commitment = ledger.commit(result["request_id"], "F1", "user_decision", "Select distinct")
    with pytest.raises(ValueError, match="committed scope"):
        execute(ledger, commitment["commitment_id"], [5, 3])
    execution = execute(ledger, commitment["commitment_id"], [4, 3])
    evidence = ledger.get_artifact(execution["evidence"]["data_hash"])
    assert evidence["scope_hash"] == commitment["scope_hash"]
    assert evidence["environment_hash"] == commitment["environment_hash"]
    assert evidence["observation_schema_hash"] == commitment["observation_schema_hash"]


def test_action_witnesses_oriented_to_intent_and_action(tmp_path):
    _, result = analyze(tmp_path)
    violated = [cell for cell in result["action_matrix"] if cell["conformance"] == "violated"]
    assert violated[0]["outcome_or_trace"]["expected"] == violated[1]["outcome_or_trace"]["actual"]
    assert violated[1]["outcome_or_trace"]["expected"] == violated[0]["outcome_or_trace"]["actual"]


def test_pending_invalid_candidates_count_toward_fork_cap(tmp_path):
    candidate = selector(True)
    candidate["typed_spec_ir"]["output"] = {"op": "python", "source": "forbidden"}
    _, result = analyze(tmp_path, records([candidate]), Budgets(max_forks=1))
    assert result["stop_reason"] == "fork_budget"
    assert result["resource_usage"]["model_calls"] == 2
    assert len(result["unresolved_candidates"]) == 1
    assert not result["forks"]


def test_unknown_fork_obligations_remain_unresolved(tmp_path):
    candidate = selector(True)
    candidate["unsupported_numeric_requirements"] = ["NaN"]
    _, result = analyze(tmp_path, records([candidate]))
    assert len(result["forks"]) == 1
    assert any("unsupported fork fields" in item.get("diagnostic", "") for item in result["unresolved_candidates"])
