"""Z3 translation and scoped evidence for the supported deterministic IR.

Unsatisfiability establishes equivalence only after a separate satisfiable
environment check. Every SAT distinguishing model is replayed by the interpreter.
"""
from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interpreter import interpret
from .ir import OBSERVATION_SCHEMA, Scope, Spec, content_hash

try:
    import z3
except ImportError:  # A missing optional solver is unresolved, never equivalence.
    z3 = None

KERNEL_VERSION = "fev-list-ir-v1"


def runtime_manifest() -> dict:
    files = ("ir.py", "interpreter.py", "solver_backend.py")
    return {"kernel": KERNEL_VERSION, "python": platform.python_version(),
            "solver": "z3", "solver_version": z3.get_version_string() if z3 else None,
            "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}}


def environment_for_scope(scope: Scope) -> dict:
    return {"version": 1, "input": "bounded_integer_list", "scope": scope.to_dict(),
            "state": "none", "assumptions": "all integer lists within scope"}


def _base(scope: Scope, specs: list[Spec], timeout_ms: int) -> dict:
    environment = environment_for_scope(scope)
    runtime = runtime_manifest()
    return {"spec_hashes": [s.hash for s in specs], "scope": scope.to_dict(), "scope_hash": scope.hash,
            "environment": environment, "environment_hash": content_hash(environment),
            "observation_schema": OBSERVATION_SCHEMA, "observation_hash": content_hash(OBSERVATION_SCHEMA),
            "runtime_manifest": runtime, "runtime_hash": content_hash(runtime), "timeout_ms": timeout_ms,
            "trust": ["IR semantics", "reference interpreter", "Z3 encoding", "Z3 solver", "observation schema"],
            "claim_scope": "bounded integer-list inputs only; does not establish natural-language intent"}


@dataclass
class _Encoded:
    kind: str
    error: Any
    scalar: Any = None
    items: list | None = None
    length: Any = None


def _empty_error():
    return z3.StringVal("")


def _cast(value: _Encoded, kind: str) -> _Encoded:
    if value.kind == kind:
        return value
    if value.kind != "error":
        raise ValueError(f"cannot coerce {value.kind} to {kind}")
    if kind == "list":
        return _Encoded(kind, value.error, items=[], length=z3.IntVal(0))
    return _Encoded(kind, value.error, scalar=z3.BoolVal(False) if kind == "bool" else z3.IntVal(0))


def _merge(condition, left: _Encoded, right: _Encoded) -> _Encoded:
    kind = right.kind if left.kind == "error" else left.kind
    left, right = _cast(left, kind), _cast(right, kind)
    error = z3.If(condition, left.error, right.error)
    if kind == "error":
        return _Encoded(kind, error)
    if kind == "list":
        capacity = max(len(left.items), len(right.items))
        litems = left.items + [z3.IntVal(0)] * (capacity - len(left.items))
        ritems = right.items + [z3.IntVal(0)] * (capacity - len(right.items))
        return _Encoded(kind, error, items=[z3.If(condition, l, r) for l, r in zip(litems, ritems)],
                        length=z3.If(condition, left.length, right.length))
    return _Encoded(kind, error, scalar=z3.If(condition, left.scalar, right.scalar))


def _values_equal(left: _Encoded, right: _Encoded):
    if left.kind != right.kind:
        return z3.BoolVal(False)
    if left.kind == "error":
        return z3.BoolVal(True)
    if left.kind != "list":
        return left.scalar == right.scalar
    size = max(len(left.items), len(right.items))
    litems = left.items + [z3.IntVal(0)] * (size - len(left.items))
    ritems = right.items + [z3.IntVal(0)] * (size - len(right.items))
    return z3.And(left.length == right.length,
                  *[z3.Implies(i < left.length, l == r) for i, (l, r) in enumerate(zip(litems, ritems))])


def _observations_equal(left: _Encoded, right: _Encoded):
    return z3.And(left.error == right.error, z3.Or(left.error != "", _values_equal(left, right)))


def _compile(spec: Spec, inputs: list, length) -> _Encoded:
    """Compile bounded lists using symbolic lengths and explicit active slots."""
    def emit(node: dict) -> _Encoded:
        op = node["op"]
        ok = _empty_error()
        if op == "input":
            return _Encoded("list", ok, items=list(inputs), length=length)
        if op in ("int", "bool"):
            constructor = z3.IntVal if op == "int" else z3.BoolVal
            return _Encoded(op, ok, scalar=constructor(node["value"]))
        if op == "list":
            return _Encoded("list", ok, items=[z3.IntVal(v) for v in node["value"]], length=z3.IntVal(len(node["value"])))
        if op == "error":
            return _Encoded("error", z3.StringVal(node["code"]))
        if op == "if":
            condition = _cast(emit(node["condition"]), "bool")
            result = _merge(condition.scalar, emit(node["then"]), emit(node["else"]))
            result.error = z3.If(condition.error != "", condition.error, result.error)
            return result
        if op == "not":
            value = _cast(emit(node["value"]), "bool")
            return _Encoded("bool", value.error, scalar=z3.Not(value.scalar))
        if op in ("sort", "unique", "length", "at"):
            values = _cast(emit(node["values"]), "list")
            if op == "length":
                return _Encoded("int", values.error, scalar=values.length)
            if op == "at":
                index = node["index"]
                error = z3.If(values.error != "", values.error,
                              z3.If(index < values.length, ok, z3.StringVal(node["error"])))
                return _Encoded("int", error, scalar=values.items[index] if index < len(values.items) else z3.IntVal(0))
            if op == "sort":
                items = list(values.items)
                # A fixed sorting network, guarded so inactive slots stay outside the list.
                for stop in range(len(items) - 1, 0, -1):
                    for i in range(stop):
                        l, r = items[i], items[i + 1]
                        swap = z3.And(i + 1 < values.length, l > r if node["order"] == "asc" else l < r)
                        items[i], items[i + 1] = z3.If(swap, r, l), z3.If(swap, l, r)
                return _Encoded("list", values.error, items=items, length=values.length)
            keeps = [z3.And(i < values.length, *[values.items[j] != value for j in range(i)])
                     for i, value in enumerate(values.items)]
            counts = [z3.Sum([z3.If(k, 1, 0) for k in keeps[:i]]) if i else z3.IntVal(0)
                      for i in range(len(keeps))]
            items = []
            for position in range(len(values.items)):
                value = z3.IntVal(0)
                for i in reversed(range(len(values.items))):
                    value = z3.If(z3.And(keeps[i], counts[i] == position), values.items[i], value)
                items.append(value)
            total = z3.Sum([z3.If(k, 1, 0) for k in keeps]) if keeps else z3.IntVal(0)
            return _Encoded("list", values.error, items=items, length=total)
        left, right = emit(node["left"]), emit(node["right"])
        if op in ("and", "or"):
            left, right = _cast(left, "bool"), _cast(right, "bool")
            use_right = left.scalar if op == "and" else z3.Not(left.scalar)
            error = z3.If(left.error != "", left.error, z3.If(use_right, right.error, ok))
            scalar = z3.And(left.scalar, right.scalar) if op == "and" else z3.Or(left.scalar, right.scalar)
            return _Encoded("bool", error, scalar=scalar)
        kind = right.kind if left.kind == "error" else left.kind
        left, right = _cast(left, kind), _cast(right, kind)
        error = z3.If(left.error != "", left.error, right.error)
        if op in ("eq", "ne"):
            equal = _values_equal(left, right)
            return _Encoded("bool", error, scalar=equal if op == "eq" else z3.Not(equal))
        left, right = _cast(left, "int"), _cast(right, "int")
        expression = {"lt": lambda: left.scalar < right.scalar, "le": lambda: left.scalar <= right.scalar,
                      "gt": lambda: left.scalar > right.scalar, "ge": lambda: left.scalar >= right.scalar}[op]()
        return _Encoded("bool", error, scalar=expression)

    data = spec.to_dict()
    result = emit(data["output"])
    signature = data["input"]
    accepted = z3.And(length >= signature["min_length"], length <= signature["max_length"],
                      *[z3.Implies(i < length, z3.And(x >= signature["items"]["min"], x <= signature["items"]["max"]))
                        for i, x in enumerate(inputs)])
    result.error = z3.If(accepted, result.error, z3.StringVal("input_out_of_bounds"))
    return result


def _environment(scope: Scope):
    length = z3.Int("input_length")
    inputs = [z3.Int(f"input_{i}") for i in range(scope.max_length)]
    constraints = [length >= scope.min_length, length <= scope.max_length]
    constraints.extend(z3.If(i < length, z3.And(x >= scope.min_value, x <= scope.max_value), x == 0)
                       for i, x in enumerate(inputs))
    return inputs, length, constraints


def _decode(model, inputs, length) -> list[int]:
    size = model.eval(length, model_completion=True).as_long()
    return [model.eval(x, model_completion=True).as_long() for x in inputs[:size]]


def _decode_observation(model, value: _Encoded) -> dict:
    error = model.eval(value.error, model_completion=True).as_string()
    if error:
        return {"kind": "error", "code": error}
    if value.kind == "list":
        result = _decode(model, value.items, value.length)
    elif value.kind == "bool":
        result = z3.is_true(model.eval(value.scalar, model_completion=True))
    else:
        result = model.eval(value.scalar, model_completion=True).as_long()
    return {"kind": "value", "value": result}


def _solver_check(solver, deadline: float):
    remaining = int((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        return "unknown", "timeout"
    solver.set(timeout=remaining)
    answer = solver.check()
    return str(answer), solver.reason_unknown() if answer == z3.unknown else None


def check_consistency(spec: Spec, scope: Scope, timeout_ms: int = 2000) -> dict:
    evidence = _base(scope, [spec], timeout_ms)
    if z3 is None:
        return {**evidence, "result": "unsupported", "status": "unresolved", "reason": "z3_solver_unavailable"}
    if type(timeout_ms) is not int or timeout_ms <= 0:
        return {**evidence, "result": "unknown", "status": "unresolved", "reason": "timeout"}
    started = time.monotonic()
    inputs, length, constraints = _environment(scope)
    solver = z3.Solver()
    solver.add(*constraints)
    encoded = _compile(spec, inputs, length)
    result, reason = _solver_check(solver, started + timeout_ms / 1000)
    evidence.update({"query_hash": content_hash(solver.to_smt2()), "result": result,
                     "query_smt2": solver.to_smt2(),
                     "status": "consistent_in_scope" if result == "sat" else "inconsistent_environment" if result == "unsat" else "unresolved"})
    if result == "sat":
        model = solver.model()
        values = _decode(model, inputs, length)
        actual, modeled = interpret(spec, values), _decode_observation(model, encoded)
        if content_hash(actual) != content_hash(modeled):
            evidence.update(result="unknown", status="unresolved", reason="encoding_replay_mismatch")
        else:
            evidence.update(witness={"input": values, "output": actual}, witness_replayed=True)
    elif reason:
        evidence["reason"] = reason
    evidence["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
    return evidence


def compare(left: Spec, right: Spec, scope: Scope, timeout_ms: int = 2000) -> dict:
    evidence = _base(scope, [left, right], timeout_ms)
    evidence.update(left_hash=left.hash, right_hash=right.hash)
    if z3 is None:
        return {**evidence, "result": "unresolved", "reason": "z3_solver_unavailable", "solver_result": "unsupported"}
    if type(timeout_ms) is not int or timeout_ms <= 0:
        return {**evidence, "result": "unresolved", "reason": "timeout", "solver_result": "unknown"}
    started = time.monotonic()
    deadline = started + timeout_ms / 1000
    inputs, length, constraints = _environment(scope)
    solver = z3.Solver()
    solver.add(*constraints)
    env_result, reason = _solver_check(solver, deadline)
    evidence.update(environment_result=env_result, environment_query_hash=content_hash(solver.to_smt2()),
                    environment_query_smt2=solver.to_smt2())
    if env_result != "sat":
        return {**evidence, "result": "unresolved", "solver_result": env_result,
                "reason": "inconsistent_environment" if env_result == "unsat" else reason}
    encoded_left, encoded_right = _compile(left, inputs, length), _compile(right, inputs, length)
    solver.add(z3.Not(_observations_equal(encoded_left, encoded_right)))
    evidence["query_hash"] = content_hash(solver.to_smt2())
    evidence["query_smt2"] = solver.to_smt2()
    result, reason = _solver_check(solver, deadline)
    evidence["solver_result"] = result
    if result == "unsat":
        evidence.update(result="equivalent_in_scope", evidence_kind="solver_unsat")
    elif result == "sat":
        model = solver.model()
        values = _decode(model, inputs, length)
        actual_left, actual_right = interpret(left, values), interpret(right, values)
        model_left, model_right = _decode_observation(model, encoded_left), _decode_observation(model, encoded_right)
        if (content_hash(actual_left) == content_hash(actual_right)
                or content_hash(actual_left) != content_hash(model_left)
                or content_hash(actual_right) != content_hash(model_right)):
            evidence.update(result="unresolved", reason="encoding_replay_mismatch")
        else:
            evidence.update(result="distinct", evidence_kind="solver_sat_replayed",
                            witness={"input": values, "left": actual_left, "right": actual_right}, witness_replayed=True)
    else:
        evidence.update(result="unresolved", reason=reason)
    evidence["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
    return evidence


def check_action(spec: Spec, action: Spec, scope: Scope, timeout_ms: int = 2000) -> dict:
    """Check a supported IR action; this makes no claim about generated code."""
    evidence = compare(spec, action, scope, timeout_ms)
    return {**evidence, "action_model_hash": action.hash, "action_model_kind": "typed_ir",
            "conformance": {"equivalent_in_scope": "checked_in_scope", "distinct": "violated",
                            "unresolved": "unknown"}[evidence["result"]]}
