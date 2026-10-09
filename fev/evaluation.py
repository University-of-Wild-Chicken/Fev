"""Versioned engineering cases; authored oracles are not human-intent labels.

Families, rather than paraphrases, form split units. This suite checks the
supported kernel and preserves unsupported cases for later model evaluations.
"""
from itertools import product
from .interpreter import interpret
from .ir import parse_and_typecheck
from .solver_backend import Scope, compare


UNSUPPORTED = [
    "Return the median rounded to two decimal places.",
    "Return the second-largest floating point value, ignoring NaNs.",
    "Delete the user and cascade removal to every related record.",
    "Permit access only if the caller has an administrator role.",
    "Return the longest string in the list.",
    "Increment every integer, wrapping on overflow.",
    "Mutate the input list in place while sorting it.",
    "Sort one million integers without a length bound.",
    "Eventually return an answer even if the server disconnects.",
    "Select a uniformly random item.",
    "Find the maximum dollar amount after converting euros at today's rate.",
    "Return a record containing the minimum, maximum, and their original positions.",
]


def engineering_cases():
    cases = []
    for order, distinct, rank, missing, bound in product(("asc", "desc"), (False, True), (1, 2, 3),
                                                       ("index_out_of_bounds", "missing_rank", "zero"), (1, 2, 3)):
        source = {"op": "input"}
        if distinct:
            source = {"op": "unique", "values": source}
        selected = {"op": "at", "index": rank - 1, "values": {"op": "sort", "order": order, "values": source},
                    "error": "index_out_of_bounds" if missing == "zero" else missing}
        output = selected
        if missing == "zero":
            output = {"op": "if", "condition": {"op": "lt", "left": {"op": "length", "values": source},
                      "right": {"op": "int", "value": rank}}, "then": {"op": "int", "value": 0}, "else": selected}
        family = f"{'distinct' if distinct else 'multiplicity'}-{order}-rank{rank}-{missing}"
        cases.append({"case_id": f"{family}-bound{bound}", "family": family,
                      "split": {1: "train", 2: "validation", 3: "test"}[rank],
                      "request": f"Select rank {rank} (one-based) in {'ascending' if order == 'asc' else 'descending'} order, "
                                 f"{'removing duplicates' if distinct else 'preserving multiplicity'}. "
                                 f"Input is a pure list of at most 3 integers in [-{bound},{bound}]. "
                                 f"If rank is missing, {'return integer zero' if missing == 'zero' else 'return error '+missing}.",
                      "supported": True, "scope": Scope(-bound, bound, 3).to_dict(),
                      "oracle": {"order": order, "distinct": distinct, "rank": rank, "missing": missing},
                      "acceptable_contracts": [{"version": 1, "input": {"kind": "list", "items": {
                          "kind": "int", "min": -bound, "max": bound}, "min_length": 0, "max_length": 3}, "output": output}],
                      "forbidden_interpretations": ["Reverse the specified ordering", "Change duplicate handling", "Change the specified missing-rank result"]})
    cases.extend({"case_id": f"unsupported-{i:02d}", "family": f"unsupported-{i:02d}", "split": "test",
                  "request": request, "supported": False, "acceptable_contracts": [],
                  "forbidden_interpretations": ["Silently discard the unsupported requirement"]}
                 for i, request in enumerate(UNSUPPORTED, 1))
    return cases


def oracle_result(case, values):
    options = case["oracle"]
    ordered = sorted(set(values) if options["distinct"] else values, reverse=options["order"] == "desc")
    if len(ordered) < options["rank"]:
        return {"kind": "value", "value": 0} if options["missing"] == "zero" else {"kind": "error", "code": options["missing"]}
    return {"kind": "value", "value": ordered[options["rank"] - 1]}


def evaluate_kernel():
    cases, count, failures = engineering_cases(), 0, []
    for case in cases:
        if not case["supported"]:
            continue
        spec = parse_and_typecheck(case["acceptable_contracts"][0])
        # Independent Python collection oracle, exhaustive in each case's scope.
        scope = Scope(**case["scope"])
        for length in range(scope.max_length + 1):
            for values in product(range(scope.min_value, scope.max_value + 1), repeat=length):
                count += 1
                actual, expected = interpret(spec, list(values)), oracle_result(case, list(values))
                if actual != expected:
                    failures.append({"case_id": case["case_id"], "input": list(values), "expected": expected, "actual": actual})
    return {"case_count": len(cases), "supported_cases": sum(c["supported"] for c in cases),
            "unsupported_reference_cases": len(UNSUPPORTED), "oracle_checks": count, "failures": failures,
            "limitations": "Engineering fixtures and an independent Python oracle; not human-intent ground truth or a model benchmark. Unsupported cases are retained, not evaluated by this kernel command."}
