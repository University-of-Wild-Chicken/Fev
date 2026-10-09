"""Only mechanically supported observations drive material questions."""
def decide(forks, comparisons, frontier, unsupported):
    valid = [f for f in forks if f["status"] == "valid"]
    ids = {f["spec_hash"]: f["fork_id"] for f in valid}
    for comparison in comparisons:
        if comparison["result"] == "distinct" and comparison.get("witness") and comparison["left_hash"] in ids and comparison["right_hash"] in ids:
            witness = comparison["witness"]
            return "clarify", {"text": f"For input {witness['input']}, which result is intended?",
                "input": witness["input"], "options": [
                    {"fork_id": ids[comparison["left_hash"]], "result": witness["left"]},
                    {"fork_id": ids[comparison["right_hash"]], "result": witness["right"]},
                    {"fork_id": None, "result": "neither / missing interpretation"}],
                "evidence_ref": comparison["evidence_ref"],
                "notice": "This example distinguishes these contracts; selecting a full contract also selects its boundary behavior and assumptions."}
    if unsupported and not valid:
        return "unsupported", {"unsupported_requirements": unsupported}
    if not valid:
        return "defer", {"reason": "No supported consistent contract retained."}
    if frontier or unsupported or any(c["result"] == "unresolved" for c in comparisons):
        return "defer", {"reason": "Material unresolved work remains; review the frontier and evidence."}
    return "defer", {"reason": "Known contracts agree in scope. No explicit selection or default policy authorizes commitment.",
                      "selectable_fork_ids": [f["fork_id"] for f in valid]}
