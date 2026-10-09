"""Execute exactly the committed IR; never generate or evaluate source code."""
from .interpreter import interpret
from .ir import Scope, content_hash, parse_and_typecheck
from .solver_backend import runtime_manifest


def execute(ledger, commitment_id, values):
    commitment = ledger.assert_current(commitment_id)
    runtime = runtime_manifest()
    checked_runtime = commitment["selected_fork"]["consistency_evidence"]["runtime_manifest"]
    for component in ("ir.py", "interpreter.py"):
        if checked_runtime.get("source_hashes", {}).get(component) != runtime["source_hashes"][component]:
            raise ValueError("committed IR runtime changed; recheck and select a new commitment before execution")
    spec = parse_and_typecheck(commitment["typed_spec_ir"])
    if spec.hash != commitment["spec_hash"]:
        raise ValueError("committed contract hash mismatch")
    scope = Scope(**commitment["scope"])
    if isinstance(values, list) and all(type(value) is int for value in values):
        if not scope.min_length <= len(values) <= scope.max_length or any(
            not scope.min_value <= value <= scope.max_value for value in values
        ):
            raise ValueError("input is outside the committed scope")
    result = interpret(spec, values)
    evidence = ledger.save_evidence(commitment["commitment_id"], {
        "evidence_kind": "reference_interpreter_execution", "spec_hash": spec.hash,
        "scope_hash": commitment["scope_hash"], "environment_hash": commitment["environment_hash"],
        "observation_schema_hash": commitment["observation_schema_hash"],
        "runtime_manifest": runtime, "runtime_hash": content_hash(runtime),
        "input": values, "result": result,
        "claim": "Execution of the committed IR. No arbitrary generated-code conformance claim."})
    return {"commitment_id": commitment["commitment_id"], "spec_hash": spec.hash,
            "result": result, "evidence": evidence}
