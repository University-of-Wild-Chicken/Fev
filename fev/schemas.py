"""Shared deterministic serialization and bounded controller configuration."""
from dataclasses import dataclass, asdict
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Budgets:
    max_forks: int = 6
    max_model_calls: int = 8
    max_generated_tokens: int = 6000
    solver_query_ms: int = 2000
    solver_total_ms: int = 30000
    max_repairs: int = 2

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_forks > 32 or self.max_model_calls > 100 or self.max_repairs > 2:
            raise ValueError("controller safety limits exceeded")


OBSERVATION = {"version": 1, "observable": "complete tagged result including errors", "mutation": "none"}
AST_GUIDE = {
    "spec": {"version": 1, "input": {"kind": "list", "items": {"kind": "int", "min": -8, "max": 8},
             "min_length": 0, "max_length": 5}, "output": {"op": "at", "index": 1,
             "values": {"op": "sort", "order": "desc", "values": {"op": "input"}}}},
    "operators": ["input", "int(value)", "bool(value)", "list(value:int[])", "error(code)",
                  "sort(values,order:asc|desc)", "unique(values)", "at(values,index:int,error?:str)",
                  "length(values)", "if(condition,then,else)", "eq/ne/lt/le/gt/ge(left,right)",
                  "and/or(left,right)", "not(value)"],
    "semantics": "at uses zero-based constant index; unique preserves first occurrence; all functions are pure. Missing index returns index_out_of_bounds. No arithmetic, strings, floats, records, state updates or arbitrary code. Mark these unsupported if required. Match input bounds to the supplied scope.",
}
