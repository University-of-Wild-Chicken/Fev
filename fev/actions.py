"""Construct IR action conformance cells from exact scoped comparisons."""
def action_matrix(forks, comparisons):
    valid = [f for f in forks if f["status"] == "valid"]
    pairs = {frozenset((c["left_hash"], c["right_hash"])): c for c in comparisons}
    cells = []
    for intended in valid:
        for action in valid:
            same = intended["spec_hash"] == action["spec_hash"]
            comparison = pairs.get(frozenset((intended["spec_hash"], action["spec_hash"])))
            if same:
                conformance, kind, refs = "checked_in_scope", "identical_ir", [intended["consistency_evidence_ref"]]
            elif comparison and comparison["result"] == "equivalent_in_scope":
                conformance, kind, refs = "checked_in_scope", "solver_ir_equivalence", [comparison["evidence_ref"]]
            elif comparison and comparison["result"] == "distinct":
                conformance, kind, refs = "violated", "replayed_distinguishing_witness", [comparison["evidence_ref"]]
            else:
                conformance, kind, refs = "unknown", "unresolved_comparison", []
            witness = comparison.get("witness") if comparison else None
            trace = None
            if witness:
                expected_side = "left" if comparison["left_hash"] == intended["spec_hash"] else "right"
                actual_side = "left" if comparison["left_hash"] == action["spec_hash"] else "right"
                trace = {"input": witness["input"], "expected": witness[expected_side], "actual": witness[actual_side]}
            cells.append({"fork_id": intended["fork_id"], "action_id": "execute_" + action["fork_id"],
                          "action_model_hash": action["spec_hash"], "conformance": conformance,
                          "evidence_kind": kind, "evidence_refs": refs,
                          "outcome_or_trace": trace,
                          "cost_components": {}, "probability": None})
        cells.append({"fork_id": intended["fork_id"], "action_id": "clarify", "action_model_hash": None,
                      "conformance": "not_applicable", "evidence_kind": "question_only",
                      "evidence_refs": [], "outcome_or_trace": None, "cost_components": {}, "probability": None})
    return cells
