"""Reference executable semantics, independent of the solver translation."""
from __future__ import annotations

from .ir import Spec


class _EvaluationError(Exception):
    def __init__(self, code: str):
        self.code = code


def interpret(spec: Spec, values: list[int]) -> dict:
    data = spec.to_dict()
    signature = data["input"]
    if not isinstance(values, list) or any(type(x) is not int for x in values):
        return {"kind": "error", "code": "invalid_input"}
    if not (signature["min_length"] <= len(values) <= signature["max_length"]) or any(
        not signature["items"]["min"] <= x <= signature["items"]["max"] for x in values
    ):
        return {"kind": "error", "code": "input_out_of_bounds"}

    def run(node: dict):
        op = node["op"]
        if op == "input":
            return list(values)
        if op in ("int", "bool"):
            return node["value"]
        if op == "list":
            return list(node["value"])
        if op == "error":
            raise _EvaluationError(node["code"])
        if op == "if":
            return run(node["then"] if run(node["condition"]) else node["else"])
        if op == "and":
            return run(node["left"]) and run(node["right"])
        if op == "or":
            return run(node["left"]) or run(node["right"])
        if op == "not":
            return not run(node["value"])
        if op in ("sort", "unique", "at", "length"):
            items = run(node["values"])
            if op == "sort":
                return sorted(items, reverse=node["order"] == "desc")
            if op == "unique":
                return list(dict.fromkeys(items))
            if op == "length":
                return len(items)
            if node["index"] >= len(items):
                raise _EvaluationError(node["error"])
            return items[node["index"]]
        left, right = run(node["left"]), run(node["right"])
        return {"eq": lambda: left == right, "ne": lambda: left != right,
                "lt": lambda: left < right, "le": lambda: left <= right,
                "gt": lambda: left > right, "ge": lambda: left >= right}[op]()

    try:
        return {"kind": "value", "value": run(data["output"])}
    except _EvaluationError as exc:
        return {"kind": "error", "code": exc.code}
