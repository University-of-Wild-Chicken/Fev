from itertools import product

import pytest
import z3

from fev.interpreter import interpret
from fev.ir import DiagnosticError, Scope, content_hash, parse_and_typecheck
from fev.solver_backend import (_compile, _decode_observation, _environment, check_action,
                                check_consistency, compare)


INPUT = {"op": "input"}


def spec(output, **bounds):
    return parse_and_typecheck({"version": 1, "input": {
        "kind": "list", "items": {"kind": "int", "min": bounds.get("min", -8), "max": bounds.get("max", 8)},
        "min_length": bounds.get("min_length", 0), "max_length": bounds.get("max_length", 5)}, "output": output})


def second(distinct=False, error="index_out_of_bounds"):
    return spec({"op": "at", "index": 1, "error": error, "values": {"op": "sort", "order": "desc",
        "values": {"op": "unique", "values": INPUT} if distinct else INPUT}})


def domains(scope):
    for length in range(scope.min_length, scope.max_length + 1):
        yield from (list(values) for values in product(range(scope.min_value, scope.max_value + 1), repeat=length))


def test_second_largest_witness_and_error_are_observable():
    assert interpret(second(True), [5, 5, 3]) == {"kind": "value", "value": 3}
    assert interpret(second(), [5, 5, 3]) == {"kind": "value", "value": 5}
    assert interpret(second(True), [5, 5]) == {"kind": "error", "code": "index_out_of_bounds"}
    result = compare(second(True), second(), Scope())
    assert result["result"] == "distinct"
    assert result["witness_replayed"] is True
    witness = result["witness"]
    assert witness["left"] == interpret(second(True), witness["input"])
    assert witness["right"] == interpret(second(), witness["input"])
    assert witness["left"] != witness["right"]
    assert compare(second(error="missing"), second(error="empty"), Scope())["result"] == "distinct"


def test_scope_and_observation_bind_evidence():
    left, right = second(True), second()
    # Length <= 1 means both return the same index error.
    evidence = compare(left, right, Scope(max_length=1))
    assert evidence["result"] == "equivalent_in_scope"
    assert evidence["solver_result"] == "unsat"
    assert evidence["environment_result"] == "sat"
    assert evidence["scope_hash"] == Scope(max_length=1).hash
    assert evidence["left_hash"] == left.hash
    for key in ("runtime_hash", "environment_hash", "observation_hash", "query_hash"):
        assert len(evidence[key]) == 64
    assert check_action(left, right, Scope(max_length=1))["conformance"] == "checked_in_scope"
    assert check_action(left, right, Scope())["conformance"] == "violated"


def test_unsatisfiable_environment_never_merges():
    empty = Scope(min_length=2, max_length=1)
    assert check_consistency(second(), empty)["result"] == "unsat"
    evidence = compare(second(), second(), empty)
    assert evidence["result"] == "unresolved"
    assert evidence["reason"] == "inconsistent_environment"
    assert compare(second(), second(), Scope(), timeout_ms=0)["result"] == "unresolved"


def test_types_and_bounds_are_observable():
    assert compare(spec({"op": "bool", "value": True}), spec({"op": "int", "value": 1}), Scope())["result"] == "distinct"
    assert interpret(second(), [True, 2])["code"] == "invalid_input"
    assert interpret(second(), [9, 2])["code"] == "input_out_of_bounds"
    assert interpret(second(), (2, 1))["code"] == "invalid_input"
    assert compare(spec(INPUT, max=1), spec(INPUT, max=2), Scope())["result"] == "distinct"


@pytest.mark.parametrize("output", [
    {"op": "python", "source": "print(1)"},
    {"op": "int", "value": True},
    {"op": "input", "ignored": "requirement"},
    {"op": "if", "condition": {"op": "int", "value": 1}, "then": INPUT, "else": INPUT},
    {"op": "sort", "values": {"op": "bool", "value": True}, "order": "asc"},
    {"op": "eq", "left": {"op": "bool", "value": True}, "right": {"op": "int", "value": 1}},
    {"op": "if", "condition": {"op": "bool", "value": True}, "then": INPUT, "else": {"op": "int", "value": 1}},
    {"op": "at", "values": INPUT, "index": -1},
    {"op": "error", "code": ""},
])
def test_rejects_unsupported_or_malformed_ast(output):
    with pytest.raises(DiagnosticError):
        spec(output)


def test_limits_and_duplicate_json_keys():
    output = INPUT
    for _ in range(17):
        output = {"op": "unique", "values": output}
    with pytest.raises(DiagnosticError, match="limit"):
        spec(output)
    with pytest.raises(DiagnosticError, match="duplicate"):
        parse_and_typecheck('{"version":1,"version":1}')
    with pytest.raises(DiagnosticError):
        spec({"op": "list", "value": list(range(17))})


def test_hash_canonical_and_immutable():
    original = second()
    copy = original.to_dict()
    copy["output"]["index"] = 0
    assert original.hash != parse_and_typecheck(copy).hash
    assert original.hash == content_hash(original.to_dict())
    assert original.to_dict()["output"]["index"] == 1


ERROR = {"op": "error", "code": "deliberate"}
ZERO = {"op": "int", "value": 0}
LENGTH = {"op": "length", "values": INPUT}
BOOLEAN = {"op": "gt", "left": LENGTH, "right": ZERO}


@pytest.mark.parametrize("expression", [
    INPUT, {"op": "sort", "values": INPUT, "order": "asc"},
    {"op": "sort", "values": INPUT, "order": "desc"},
    {"op": "unique", "values": INPUT},
    {"op": "unique", "values": {"op": "sort", "values": INPUT, "order": "desc"}},
    {"op": "at", "values": INPUT, "index": 0},
    {"op": "at", "values": {"op": "unique", "values": INPUT}, "index": 1, "error": "missing"},
    {"op": "length", "values": {"op": "unique", "values": INPUT}},
    {"op": "if", "condition": BOOLEAN, "then": INPUT, "else": {"op": "list", "value": [2, 3]}},
    {"op": "if", "condition": BOOLEAN, "then": ZERO, "else": ERROR},
    {"op": "if", "condition": ERROR, "then": INPUT, "else": {"op": "list", "value": []}},
    {"op": "and", "left": BOOLEAN, "right": ERROR},
    {"op": "or", "left": BOOLEAN, "right": ERROR},
    {"op": "not", "value": BOOLEAN},
    {"op": "eq", "left": INPUT, "right": {"op": "unique", "values": INPUT}},
    {"op": "ne", "left": INPUT, "right": {"op": "list", "value": [0]}},
    {"op": "lt", "left": LENGTH, "right": {"op": "int", "value": 2}},
    {"op": "le", "left": LENGTH, "right": {"op": "int", "value": 2}},
    {"op": "ge", "left": LENGTH, "right": {"op": "int", "value": 2}},
    {"op": "eq", "left": ERROR, "right": ERROR},
    {"op": "at", "values": ERROR, "index": 0},
    {"op": "length", "values": ERROR},
    {"op": "sort", "values": ERROR, "order": "asc"},
    {"op": "if", "condition": BOOLEAN, "then": ERROR, "else": ERROR},
])
def test_exhaustive_interpreter_solver_translation_agreement(expression):
    scope = Scope(min_value=-1, max_value=1, max_length=3)
    contract = spec(expression, min=-1, max=1, max_length=3)
    inputs, length, constraints = _environment(scope)
    encoded = _compile(contract, inputs, length)
    solver = z3.Solver()
    solver.add(*constraints)
    for values in domains(scope):
        solver.push()
        solver.add(length == len(values), *[inputs[i] == x for i, x in enumerate(values)])
        assert solver.check() == z3.sat
        modeled = _decode_observation(solver.model(), encoded)
        assert content_hash(modeled) == content_hash(interpret(contract, values)), (expression, values, modeled)
        solver.pop()


def test_exhaustive_input_rejection_translation_agreement():
    scope = Scope(min_value=-1, max_value=1, max_length=3)
    contract = spec(INPUT, min=-1, max=0, max_length=2)
    inputs, length, constraints = _environment(scope)
    encoded = _compile(contract, inputs, length)
    solver = z3.Solver()
    solver.add(*constraints)
    for values in domains(scope):
        solver.push()
        solver.add(length == len(values), *[inputs[i] == x for i, x in enumerate(values)])
        assert solver.check() == z3.sat
        assert _decode_observation(solver.model(), encoded) == interpret(contract, values)
        solver.pop()


def test_replay_mismatch_and_missing_solver_fail_closed(monkeypatch):
    import fev.solver_backend as backend
    monkeypatch.setattr(backend, "interpret", lambda *_: {"kind": "value", "value": 99})
    assert compare(second(), second(True), Scope())["reason"] == "encoding_replay_mismatch"
    assert check_consistency(second(), Scope())["reason"] == "encoding_replay_mismatch"
    monkeypatch.setattr(backend, "z3", None)
    assert compare(second(), second(), Scope())["result"] == "unresolved"
    assert check_consistency(second(), Scope())["result"] == "unsupported"


def test_comparator_agrees_with_exhaustive_behavior_classes():
    scope = Scope(min_value=0, max_value=1, max_length=3)
    candidates = [second(), second(True), spec(INPUT), spec({"op": "unique", "values": INPUT}),
                  spec({"op": "sort", "values": INPUT, "order": "asc"}),
                  spec({"op": "sort", "values": {"op": "sort", "values": INPUT, "order": "asc"}, "order": "asc"}),
                  spec(ERROR), spec(ZERO)]
    for i, left in enumerate(candidates):
        for right in candidates[i:]:
            differs = any(content_hash(interpret(left, values)) != content_hash(interpret(right, values)) for values in domains(scope))
            evidence = compare(left, right, scope)
            assert evidence["result"] == ("distinct" if differs else "equivalent_in_scope")
