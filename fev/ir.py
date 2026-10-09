"""Strict, bounded list-selection IR. JSON transports data; it never transports code."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

IR_VERSION = 1
MAX_NODES = 128
MAX_DEPTH = 16
MAX_LIST_LENGTH = 16
MAX_INTEGER = 2**31 - 1
OBSERVATION_SCHEMA = {"version": 1, "success": "typed JSON value", "failure": "error code", "effects": "pure"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class DiagnosticError(ValueError):
    def __init__(self, message: str, *, path: str = "$", code: str = "invalid_ir"):
        self.diagnostics = [{"code": code, "path": path, "message": message}]
        self.code = code
        super().__init__(f"{path}: {message}")

    def to_dict(self) -> dict:
        return {"diagnostics": self.diagnostics}


def _keys(data: Any, required: set[str], optional: set[str], path: str) -> None:
    if not isinstance(data, dict):
        raise DiagnosticError("expected an object", path=path)
    if not all(isinstance(key, str) for key in data):
        raise DiagnosticError("object keys must be strings", path=path)
    missing, extra = required - data.keys(), data.keys() - required - optional
    if missing or extra:
        raise DiagnosticError(f"missing keys {sorted(missing)}; unsupported keys {sorted(extra)}", path=path)


def _integer(value: Any, path: str, low: int = -MAX_INTEGER, high: int = MAX_INTEGER) -> int:
    if type(value) is not int or not low <= value <= high:
        raise DiagnosticError(f"expected integer in [{low}, {high}]", path=path)
    return value


def _error_code(value: Any, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise DiagnosticError("expected a lowercase error identifier (1–64 characters)", path=path)
    return value


@dataclass(frozen=True)
class Spec:
    """Immutable canonical bytes prevent mutation after an evidence hash is issued."""
    _canonical: str
    output_type: str

    def to_dict(self) -> dict:
        return json.loads(self._canonical)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self._canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Scope:
    min_value: int = -8
    max_value: int = 8
    max_length: int = 5
    min_length: int = 0

    def __post_init__(self) -> None:
        _integer(self.min_value, "scope.min_value")
        _integer(self.max_value, "scope.max_value")
        _integer(self.min_length, "scope.min_length", 0, MAX_LIST_LENGTH)
        _integer(self.max_length, "scope.max_length", 0, MAX_LIST_LENGTH)

    def to_dict(self) -> dict:
        return {"min_value": self.min_value, "max_value": self.max_value,
                "min_length": self.min_length, "max_length": self.max_length}

    @property
    def hash(self) -> str:
        return content_hash(self.to_dict())


def parse_and_typecheck(data: Any) -> Spec:
    """Validate all fields, operand types and resource limits; normalize defaults."""
    if isinstance(data, str):
        if len(data) > 65536:
            raise DiagnosticError("JSON text exceeds 65536 characters", code="resource_limit")
        def no_duplicates(pairs: list) -> dict:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise DiagnosticError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result
        try:
            data = json.loads(data, object_pairs_hook=no_duplicates)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise DiagnosticError("invalid JSON") from exc
    _keys(data, {"version", "input", "output"}, set(), "$")
    if type(data["version"]) is not int or data["version"] != IR_VERSION:
        raise DiagnosticError("unsupported IR version", path="$.version", code="unsupported")
    signature = data["input"]
    _keys(signature, {"kind", "items", "max_length"}, {"min_length"}, "$.input")
    if signature["kind"] != "list":
        raise DiagnosticError("only integer-list inputs are supported", path="$.input.kind", code="unsupported")
    item = signature["items"]
    _keys(item, {"kind", "min", "max"}, set(), "$.input.items")
    if item["kind"] != "int":
        raise DiagnosticError("only bounded integer items are supported", path="$.input.items.kind", code="unsupported")
    lo = _integer(item["min"], "$.input.items.min")
    hi = _integer(item["max"], "$.input.items.max")
    minimum = _integer(signature.get("min_length", 0), "$.input.min_length", 0, MAX_LIST_LENGTH)
    maximum = _integer(signature["max_length"], "$.input.max_length", 0, MAX_LIST_LENGTH)
    if lo > hi or minimum > maximum:
        raise DiagnosticError("input bounds are inconsistent", path="$.input")
    node_count = 0

    def visit(node: Any, path: str, depth: int) -> tuple[dict, str]:
        nonlocal node_count
        node_count += 1
        if depth > MAX_DEPTH or node_count > MAX_NODES:
            raise DiagnosticError("AST depth or node limit exceeded", path=path, code="resource_limit")
        if not isinstance(node, dict) or not isinstance(node.get("op"), str):
            raise DiagnosticError("expression requires a string op", path=path)
        op = node["op"]
        fields = {
            "input": (), "int": ("value",), "bool": ("value",), "list": ("value",),
            "error": ("code",), "sort": ("values", "order"), "unique": ("values",),
            "at": ("values", "index"), "length": ("values",),
            "if": ("condition", "then", "else"), "not": ("value",),
            **{name: ("left", "right") for name in ("eq", "ne", "lt", "le", "gt", "ge", "and", "or")},
        }
        if op not in fields:
            raise DiagnosticError(f"unsupported operator {op!r}", path=path, code="unsupported")
        _keys(node, {"op", *fields[op]}, {"error"} if op == "at" else set(), path)
        clean = {"op": op}

        def child(key: str, expected: str | None = None) -> str:
            value, kind = visit(node[key], f"{path}.{key}", depth + 1)
            if expected is not None and kind not in (expected, "error"):
                raise DiagnosticError(f"expected {expected}, got {kind}", path=f"{path}.{key}", code="type_error")
            clean[key] = value
            return kind

        if op == "input":
            kind = "list"
        elif op == "int":
            clean["value"] = _integer(node["value"], f"{path}.value")
            kind = "int"
        elif op == "bool":
            if type(node["value"]) is not bool:
                raise DiagnosticError("expected boolean", path=f"{path}.value", code="type_error")
            clean["value"] = node["value"]
            kind = "bool"
        elif op == "list":
            if not isinstance(node["value"], list) or len(node["value"]) > MAX_LIST_LENGTH:
                raise DiagnosticError("expected bounded integer list", path=f"{path}.value")
            clean["value"] = [_integer(x, f"{path}.value[{i}]") for i, x in enumerate(node["value"])]
            kind = "list"
        elif op == "error":
            clean["code"] = _error_code(node["code"], f"{path}.code")
            kind = "error"
        elif op in ("sort", "unique", "at", "length"):
            child("values", "list")
            kind = "list" if op in ("sort", "unique") else "int"
            if op == "sort":
                if node["order"] not in ("asc", "desc"):
                    raise DiagnosticError("order must be asc or desc", path=f"{path}.order")
                clean["order"] = node["order"]
            if op == "at":
                clean["index"] = _integer(node["index"], f"{path}.index", 0, MAX_LIST_LENGTH)
                clean["error"] = _error_code(node.get("error", "index_out_of_bounds"), f"{path}.error")
        elif op == "if":
            child("condition", "bool")
            then, otherwise = child("then"), child("else")
            if then != otherwise and "error" not in (then, otherwise):
                raise DiagnosticError("if branches must have the same type", path=path, code="type_error")
            kind = otherwise if then == "error" else then
        elif op == "not":
            child("value", "bool")
            kind = "bool"
        else:
            expected = "bool" if op in ("and", "or") else "int" if op in ("lt", "le", "gt", "ge") else None
            left, right = child("left", expected), child("right", expected)
            if left != right and "error" not in (left, right):
                raise DiagnosticError("operands must have the same type", path=path, code="type_error")
            kind = "bool"
        return clean, kind

    output, output_type = visit(data["output"], "$.output", 1)
    normalized = {"version": IR_VERSION, "input": {"kind": "list", "items": {"kind": "int", "min": lo, "max": hi},
                  "min_length": minimum, "max_length": maximum}, "output": output}
    return Spec(canonical_json(normalized), output_type)
