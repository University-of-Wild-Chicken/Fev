#!/usr/bin/env python3
"""Generate source-bound Fev SFT examples with exhaustive synthetic oracles.

All labels concern authored synthetic contracts and finite formal properties.
They are not human intent annotations, calibration targets, or accuracy evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fev.interpreter import interpret
from fev.ir import Scope, canonical_json, content_hash, parse_and_typecheck
from fev.model_adapter import SYSTEM_PROMPT, proposal_schema
from fev.schemas import AST_GUIDE
from fev.solver_backend import compare

GENERATOR_VERSION = "fev-synthetic-v1.2"
FAMILIES = {
    "rank_duplicate_policy": "train",
    "sort_direction": "train",
    "unique_order": "train",
    "count_duplicate_policy": "train",
    "position_or_rank": "train",
    "length_boundary": "train",
    "extremum_direction": "train",
    "empty_result_policy": "train",
    "guarded_rank": "validation",
    "extremum_threshold": "validation",
    "compound_predicate": "test",
    "conditional_projection": "test",
    "unsupported_strings": "train",
    "unsupported_floats": "train",
    "unsupported_state": "validation",
    "unsupported_filesystem": "validation",
    "unsupported_temporal": "test",
    "unsupported_randomness": "test",
}


def node(op, **kwargs):
    return {"op": op, **kwargs}


def constant(value):
    return node("bool" if type(value) is bool else "int", value=value)


def length(values=None):
    return node("length", values=values or node("input"))


def ordered(order="asc", distinct=False):
    values = node("unique", values=node("input")) if distinct else node("input")
    return node("sort", values=values, order=order)


def select(values, index=0):
    return node("at", values=values, index=index, error="index_out_of_bounds")


def condition(op, left, right):
    return node(op, left=left, right=right)


def branch(condition_, then, else_):
    return node("if", condition=condition_, then=then, **{"else": else_})


def result(value):
    return {"kind": "value", "value": value}


def at_or_error(values, index=0):
    if index >= len(values):
        return {"kind": "error", "code": "index_out_of_bounds"}
    return result(values[index])


def parameters(index):
    return {"rank": 1 + index % 2, "boundary": 1 + index % 3,
            "threshold": index % 3 - 1, "sentinel": 10 + index,
            "scope": Scope(min_value=-2 - (index % 2), max_value=2 + ((index // 2) % 2),
                           min_length=0, max_length=3 + ((index // 4) % 2))}


def family_case(family, p):
    """Author two ASTs and natural-language conventions; no model labels."""
    k, n, t, sentinel = p["rank"], p["boundary"], p["threshold"], p["sentinel"]
    inp = node("input")
    size, distinct_size = length(), length(node("unique", values=inp))
    max_value = select(ordered("desc"))
    cases = {
        "rank_duplicate_policy": (
            f"Return the number at descending rank {k + 1}.", "How should duplicate values affect rank?",
            ("Count each occurrence when ranking.", "Rank distinct values only."),
            (select(ordered("desc"), k), select(ordered("desc", True), k))),
        "sort_direction": (
            "Return the integers in sorted order.", "Which sort direction is intended?",
            ("Use ascending numeric order.", "Use descending numeric order."),
            (ordered("asc"), ordered("desc"))),
        "unique_order": (
            "Return the unique integers as a list.", "How should unique values be ordered?",
            ("Preserve first occurrence order.", "Sort distinct values in ascending order."),
            (node("unique", values=inp), ordered("asc", True))),
        "count_duplicate_policy": (
            "Count how many integers the list contains.", "Does the count include duplicate occurrences?",
            ("Count every occurrence.", "Count distinct values only."), (size, distinct_size)),
        "position_or_rank": (
            f"Return integer number {k + 1} from the list.", "Does numbering follow input order or numeric order?",
            ("Number entries in their original input order.", "Number entries in ascending numeric order."),
            (select(inp, k), select(ordered(), k))),
        "length_boundary": (
            f"Check the list length against a size limit of {n}.", "Is the boundary included in the test?",
            (f"Return true only when length is greater than {n}.", f"Return true when length is at least {n}."),
            (condition("gt", size, constant(n)), condition("ge", size, constant(n)))),
        "extremum_direction": (
            "Return an extreme integer from the list.", "Which numeric extreme is intended?",
            ("Return the minimum integer.", "Return the maximum integer."),
            (select(ordered("asc")), max_value)),
        "empty_result_policy": (
            "Return the first input integer.", "What should an empty input produce?",
            ("Return the error index_out_of_bounds for empty input.", f"Return integer {sentinel} for empty input."),
            (select(inp), branch(condition("eq", size, constant(0)), constant(sentinel), select(inp)))),
        "guarded_rank": (
            f"Return the minimum if at least {n} values are present; otherwise return {sentinel}.",
            "Does the minimum's guard count occurrences or distinct values?",
            ("Count occurrences for the size guard.", "Count distinct values for the size guard."),
            (branch(condition("ge", size, constant(n)), select(ordered()), constant(sentinel)),
             branch(condition("ge", distinct_size, constant(n)), select(ordered()), constant(sentinel)))),
        "extremum_threshold": (
            f"Check whether the maximum integer meets threshold {t}.", "Does meeting the threshold include equality?",
            (f"Require the maximum to be greater than {t}.", f"Require the maximum to be at least {t}."),
            (condition("gt", max_value, constant(t)), condition("ge", max_value, constant(t)))),
        "compound_predicate": (
            f"Decide acceptance using these two criteria: length at least {n}; maximum at least {t}.",
            "Should both criteria or either criterion suffice?",
            ("Both criteria must hold; evaluate the length test first.",
             "Either criterion may hold; evaluate the length test first."),
            (condition("and", condition("ge", size, constant(n)), condition("ge", max_value, constant(t))),
             condition("or", condition("ge", size, constant(n)), condition("ge", max_value, constant(t))))),
        "conditional_projection": (
            f"If there are at least {n} values, return them sorted; otherwise return [{sentinel}].",
            "Does the branch condition count occurrences or distinct values?",
            ("Count occurrences before choosing the branch; sort ascending.",
             "Count distinct values before choosing the branch; sort ascending."),
            (branch(condition("ge", size, constant(n)), ordered(), node("list", value=[sentinel])),
             branch(condition("ge", distinct_size, constant(n)), ordered(), node("list", value=[sentinel])))),
    }
    return cases[family]


def oracle(family, choice, values, p):
    """Independent direct Python functions; deliberately never walk an AST."""
    k, n, t, sentinel = p["rank"], p["boundary"], p["threshold"], p["sentinel"]
    if family == "rank_duplicate_policy":
        return at_or_error(sorted(values if choice == 0 else set(values), reverse=True), k)
    if family == "sort_direction":
        return result(sorted(values, reverse=bool(choice)))
    if family == "unique_order":
        if choice:
            return result(sorted(set(values)))
        seen = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return result(seen)
    if family == "count_duplicate_policy":
        return result(len(values) if choice == 0 else len(set(values)))
    if family == "position_or_rank":
        return at_or_error(values if choice == 0 else sorted(values), k)
    if family == "length_boundary":
        return result(len(values) > n if choice == 0 else len(values) >= n)
    if family == "extremum_direction":
        return result(min(values) if choice == 0 else max(values)) if values else at_or_error([])
    if family == "empty_result_policy":
        return at_or_error(values) if choice == 0 or values else result(sentinel)
    if family == "guarded_rank":
        size = len(values) if choice == 0 else len(set(values))
        return result(min(values)) if size >= n else result(sentinel)
    if family == "extremum_threshold":
        if not values:
            return at_or_error([])
        return result(max(values) > t if choice == 0 else max(values) >= t)
    if family == "compound_predicate":
        large = len(values) >= n
        if choice == 0 and not large:
            return result(False)
        if choice == 1 and large:
            return result(True)
        return result(max(values) >= t) if values else at_or_error([])
    if family == "conditional_projection":
        size = len(values) if choice == 0 else len(set(values))
        return result(sorted(values) if size >= n else [sentinel])
    raise ValueError(f"Unknown family: {family}")


def domain(scope):
    for size in range(scope.min_length, scope.max_length + 1):
        for values in itertools.product(range(scope.min_value, scope.max_value + 1), repeat=size):
            yield list(values)


def spec_for(output, scope):
    return {"version": 1, "input": {"kind": "list", "items": {
        "kind": "int", "min": scope.min_value, "max": scope.max_value},
        "min_length": scope.min_length, "max_length": scope.max_length}, "output": output}


def equivalent_form(spec, variant):
    output = deepcopy(spec["output"])
    cond = (constant(True), condition("ge", length(), constant(0)),
            condition("eq", length(), length()),
            node("not", value=constant(False)))[variant % 4]
    return {**deepcopy(spec), "output": branch(cond, output, node("error", code="unreachable"))}


def validate_schema(value, schema, root=None):
    """Validate exactly the schema subset emitted by proposal_schema."""
    root = schema if root is None else root
    if "$ref" in schema:
        ref = root
        for part in schema["$ref"].split("/")[1:]:
            ref = ref[part]
        return validate_schema(value, ref, root)
    if "oneOf" in schema:
        matches = 0
        for candidate in schema["oneOf"]:
            try:
                validate_schema(value, candidate, root)
                matches += 1
            except ValueError:
                pass
        if matches != 1:
            raise ValueError("Target must match exactly one expression form")
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}
    if expected and type(value) is not types[expected]:
        raise ValueError(f"Expected {expected}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Target enum mismatch")
    if "const" in schema and value != schema["const"]:
        raise ValueError("Target constant mismatch")
    if expected == "object":
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise ValueError("Missing target fields")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], root)
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"Unexpected target field: {key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate_schema(item, schema["additionalProperties"], root)
    if expected == "array":
        if len(value) > schema.get("maxItems", len(value)):
            raise ValueError("Too many target array items")
        for item in value:
            validate_schema(item, schema["items"], root)


def validate_record(record):
    messages = record["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Invalid message roles")
    user, target = json.loads(messages[1]["content"]), json.loads(messages[2]["content"])
    validate_schema(target, proposal_schema(user["stage"]))
    source = user["packet"]["raw_text"]
    for claim in target.get("claims", []):
        if claim["provenance"] == "explicit_text" and source[claim["start"]:claim["end"]] != claim["text"]:
            raise ValueError("Source claim span mismatch")
    specs = []
    for fork in target.get("forks", []):
        specs.append(parse_and_typecheck(fork["typed_spec_ir"]))
        for assumption in fork["assumptions"]:
            if assumption["provenance"] == "explicit_text":
                if source[assumption["start"]:assumption["end"]] != assumption["text"]:
                    raise ValueError("Assumption span mismatch")
    return specs


def verify_parent(family, index):
    p = parameters(index)
    # Guarded distinct counting needs threshold > 1 to create a material pair.
    if family in ("guarded_rank", "conditional_projection"):
        p["boundary"] = 2 + index % 2
    _, _, _, expressions = family_case(family, p)
    specs = [parse_and_typecheck(spec_for(expr, p["scope"])) for expr in expressions]
    duplicate = parse_and_typecheck(equivalent_form(specs[0].to_dict(), index))
    first_witness = None
    count = 0
    for values in domain(p["scope"]):
        outputs = [oracle(family, choice, values, p) for choice in (0, 1)]
        for choice in (0, 1):
            if interpret(specs[choice], values) != outputs[choice]:
                raise ValueError(f"Independent oracle mismatch: {family}/{index}/{values}/{choice}")
        if interpret(duplicate, values) != outputs[0]:
            raise ValueError("Duplicate rewrite changes semantics")
        if first_witness is None and outputs[0] != outputs[1]:
            first_witness = {"input": values, "left": outputs[0], "right": outputs[1]}
        count += 1
    if first_witness is None:
        raise ValueError(f"Synthetic material pair is equivalent: {family}/{index}")
    material = compare(*specs, p["scope"], timeout_ms=2000)
    duplicate_evidence = compare(specs[0], duplicate, p["scope"], timeout_ms=2000)
    if material["result"] != "distinct" or duplicate_evidence["result"] != "equivalent_in_scope":
        raise ValueError(f"Solver did not validate synthetic labels: {family}/{index}")
    for choice in (0, 1):
        actual = oracle(family, choice, material["witness"]["input"], p)
        if actual != material["witness"]["left" if choice == 0 else "right"]:
            raise ValueError("Solver witness fails independent oracle replay")
    material.pop("elapsed_ms", None)
    duplicate_evidence.pop("elapsed_ms", None)
    evidence = {"parent_id": f"{family}/{index:03d}", "scope": p["scope"].to_dict(),
                "oracle_inputs_checked": count, "oracle_evaluations": count * 3,
                "canonical_witness": first_witness, "material": material,
                "duplicate": duplicate_evidence, "specs": [s.to_dict() for s in specs],
                "equivalent_spec": duplicate.to_dict()}
    return p, evidence


def make_records(family, index, surface, p, evidence):
    task, question, meanings, _ = family_case(family, p)
    prefixes = ("", "For an input list of integers, ", "List operation: ", "Please ")
    source = prefixes[surface] + (task if surface == 0 else task[0].lower() + task[1:])
    error_convention = "When an index is unavailable, return the error index_out_of_bounds."
    explicit_error = family in {"rank_duplicate_policy", "position_or_rank", "extremum_direction",
                                "guarded_rank", "extremum_threshold", "compound_predicate"}
    if explicit_error:
        source += " " + error_convention
    resolved = surface == 3
    selected = index % 2
    if resolved:
        source += " " + meanings[selected]
    axis = {"axis_id": family, "question": question, "options": [
        {"option_id": f"option_{i}", "meaning": meaning} for i, meaning in enumerate(meanings)],
        "resolved_by_source": resolved}
    claim = {"text": source, "start": 0, "end": len(source), "formal_meaning": task,
             "provenance": "explicit_text"}
    base = {"raw_text": source, "scope": p["scope"].to_dict(), "context": {}, "language": AST_GUIDE}
    forks = []
    for choice in (0, 1):
        chosen = {"text": meanings[choice], "provenance": "inferred"}
        if resolved and choice == selected:
            start = source.index(meanings[choice])
            chosen.update(provenance="explicit_text", start=start, end=start + len(meanings[choice]))
        assumptions = [chosen, {"text": "The declared finite input scope is an analysis assumption.", "provenance": "external_assumption"}]
        if explicit_error:
            start = source.index(error_convention)
            assumptions.append({"text": error_convention, "provenance": "explicit_text",
                                "start": start, "end": start + len(error_convention)})
        forks.append({"label": meanings[choice], "selected_axis_options": {family: f"option_{choice}"},
                      "assumptions": assumptions,
                      "typed_spec_ir": evidence["specs"][choice], "unresolved_axes": [],
                      "unsupported_requirements": []})
    target_forks = [forks[selected]] if resolved else forks
    # A general absence of a completeness proof is not a concrete search item.
    frontier = []
    tasks = [("intake", base, {"claims": [claim], "ambiguity_axes": [axis], "unsupported_requirements": []})]
    tasks.append(("proposal", {**base, "claims": [claim], "ambiguity_axes": [axis]},
                  {"forks": target_forks, "search_frontier": frontier, "unsupported_requirements": []}))
    damaged = deepcopy(target_forks[0])
    if surface % 2:
        damaged["typed_spec_ir"]["version"] = "1"
    else:
        damaged["typed_spec_ir"]["output"]["op"] += "_misspelled"
    try:
        parse_and_typecheck(damaged["typed_spec_ir"])
    except ValueError as exc:
        diagnostic = str(exc)
    else:
        raise ValueError("Syntax-repair input was not rejected")
    tasks.append(("repair", {**base, "compiler_diagnostic": {"kind": "compiler_diagnostic",
                 "candidate": damaged, "diagnostic": diagnostic, "stage": "proposal"},
                 "instruction": "Repair only the syntax/type error, preserving the candidate's stated choice and source meaning."},
                 {"forks": [target_forks[0]], "search_frontier": frontier, "unsupported_requirements": []}))
    duplicate = surface % 2 == 0
    comparison = evidence["duplicate"] if duplicate else evidence["material"]
    relation = "equivalent_in_scope" if duplicate else "distinct"
    note = ("Recorded kernel evidence reports equivalent_in_scope for the two contracts; they are semantic duplicates within the declared scope. This does not establish user intent." if duplicate else
            "Recorded kernel evidence reports distinct contracts. The independently replayed witness changes observable behavior; ask about that result if the source leaves the choice unresolved.")
    right = evidence["equivalent_spec"] if duplicate else evidence["specs"][1]
    critic_packet = {**base, "claims": [claim], "ambiguity_axes": [axis],
                    "retained": [{"fork_id": "left", "typed_spec_ir": evidence["specs"][0], "status": "valid"},
                                 {"fork_id": "right", "typed_spec_ir": right, "status": "valid"}],
                    "search_frontier": frontier, "unsupported_requirements": [],
                    "comparisons": [{key: comparison[key] for key in (
                         "result", "left_hash", "right_hash", "scope_hash", "witness") if key in comparison}]}
    tasks.append(("critic", critic_packet, {"critic_notes": [note], "search_frontier": frontier,
                                            "unsupported_requirements": []}))
    records = []
    for stage, packet, target in tasks:
        record = {"id": f"{family}-{index:03d}-surface{surface}-{stage}", "parent_id": evidence["parent_id"],
                  "family": family, "split": FAMILIES[family], "stage": stage,
                  "label_kind": "synthetic_contract_property_not_human_intent",
                  "formal_relation": relation if stage == "critic" else None,
                  "evidence_ref": "evidence.jsonl#" + evidence["parent_id"],
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": canonical_json({"stage": stage, "packet": packet})},
                               {"role": "assistant", "content": canonical_json(target)}]}
        validate_record(record)
        records.append(record)
    return records


def unsupported_records(family, count=8):
    patterns = {
        "unsupported_strings": ("Lowercase each of {n} input strings.", "String inputs and case conversion are outside the integer-list language."),
        "unsupported_floats": ("Compute the mean of decimal input numbers to {n} fractional digits.", "Floating-point input, arithmetic means and decimal rounding are unsupported."),
        "unsupported_state": ("Increment a persistent counter {n} times and save its new value.", "Persistent mutable state and state updates are outside this pure language."),
        "unsupported_filesystem": ("Delete {n} files supplied as input paths.", "Path strings, filesystem access and deletion effects are unsupported."),
        "unsupported_temporal": ("Wait {n} seconds after the input arrives, then return the current time.", "Temporal behavior, waiting and clock observations are unsupported."),
        "unsupported_randomness": ("Return {n} randomly sampled integers from the input list.", "Random sampling and probabilistic observations are unsupported."),
    }
    pattern, limitation = patterns[family]
    records = []
    for index in range(count):
        source = pattern.format(n=index + 1)
        packet = {"raw_text": source, "scope": Scope().to_dict(), "context": {}, "language": AST_GUIDE}
        claim = {"text": source, "start": 0, "end": len(source), "formal_meaning": source,
                 "provenance": "explicit_text"}
        for stage in ("intake", "proposal"):
            target = ({"claims": [claim], "ambiguity_axes": [], "unsupported_requirements": [limitation]}
                      if stage == "intake" else {"forks": [], "search_frontier": ["No complete contract can be represented by the current domain adapter."], "unsupported_requirements": [limitation]})
            row = {"id": f"{family}-{index:03d}-{stage}", "parent_id": f"{family}/{index:03d}",
                   "family": family, "split": FAMILIES[family], "stage": stage,
                   "label_kind": "synthetic_capability_boundary_not_human_intent", "formal_relation": None,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": canonical_json({"stage": stage, "packet": packet})},
                                {"role": "assistant", "content": canonical_json(target)}]}
            validate_record(row)
            records.append(row)
    return records


def all_spec_outputs(record):
    """Fingerprints include prompt-side critic/candidate contracts, ignoring scope."""
    packet = json.loads(record["messages"][1]["content"])
    target = json.loads(record["messages"][2]["content"])
    hashes = set()
    def walk(value):
        if isinstance(value, dict):
            if set(value) == {"version", "input", "output"}:
                try:
                    spec = parse_and_typecheck(value)
                    hashes.add(content_hash(spec.to_dict()["output"]))
                except ValueError:
                    pass  # Intentionally malformed repair source, checked separately.
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    # Exclude the shared language guide, which is an identical schema example
    # in every split and is not an authored contract for that task.
    packet.get("packet", {}).pop("language", None)
    walk(packet)
    walk(target)
    return hashes


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output, parents_per_family=16, surfaces=4):
    if not 1 <= parents_per_family <= 64 or not 1 <= surfaces <= 4:
        raise ValueError("Use 1..64 parents per family and 1..4 surfaces")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; preserve the existing frozen dataset")
    output.mkdir(parents=True, exist_ok=True)
    records = {split: [] for split in ("train", "validation", "test")}
    evidence_rows = []
    duplicate_parents_removed = 0
    for family, split in FAMILIES.items():
        if family.startswith("unsupported_"):
            records[split].extend(unsupported_records(family))
            continue
        seen_parents = set()
        for index in range(parents_per_family):
            p, evidence = verify_parent(family, index)
            task, question, meanings, _ = family_case(family, p)
            identity = content_hash([task, question, meanings, evidence["specs"]])
            if identity in seen_parents:
                duplicate_parents_removed += 1
                continue
            seen_parents.add(identity)
            evidence_rows.append(evidence)
            for surface in range(surfaces):
                records[split].extend(make_records(family, index, surface, p, evidence))
        print(f"Validated {family}: {len(seen_parents)} unique authored parents", file=sys.stderr, flush=True)
    duplicate_records_removed = 0
    for split, rows in records.items():
        unique = {}
        for row in rows:
            identity = content_hash(row["messages"])
            if identity in unique:
                duplicate_records_removed += 1
            else:
                unique[identity] = row
        records[split] = list(unique.values())
    hashes = {}
    for split, rows in records.items():
        data = "".join(canonical_json(row) + "\n" for row in rows)
        path = output / (split + ".jsonl")
        path.write_text(data, encoding="utf-8")
        hashes[path.name] = sha_file(path)
    evidence_path = output / "evidence.jsonl"
    evidence_path.write_text("".join(canonical_json(row) + "\n" for row in evidence_rows))
    hashes[evidence_path.name] = sha_file(evidence_path)
    parents = {split: sorted({row["parent_id"] for row in rows}) for split, rows in records.items()}
    spec_sets = {split: {spec.hash for row in rows for spec in validate_record(row)}
                 for split, rows in records.items()}
    output_sets = {split: set().union(*(all_spec_outputs(row) for row in rows)) for split, rows in records.items()}
    for left, right in itertools.combinations(records, 2):
        if set(parents[left]) & set(parents[right]) or spec_sets[left] & spec_sets[right]:
            raise ValueError("Parent or exact contract leaked across dataset splits")
        if output_sets[left] & output_sets[right]:
            raise ValueError("Scope-independent authored output AST leaked across dataset splits")
    summary = {
        "status": "passed", "records_validated": sum(map(len, records.values())),
        "parent_tasks": sum(map(len, parents.values())), "supported_parent_tasks": len(evidence_rows), "solver_material_pairs": len(evidence_rows),
        "solver_equivalent_pairs": len(evidence_rows),
        "independent_oracle_inputs": sum(e["oracle_inputs_checked"] for e in evidence_rows),
        "independent_oracle_evaluations": sum(e["oracle_evaluations"] for e in evidence_rows),
        "solver_witnesses_replayed_by_independent_oracles": len(evidence_rows),
        "source_spans_checked": True, "target_json_schema_checked": True,
        "every_target_spec_typechecked": True, "repair_inputs_rejected_before_correction": True,
        "cross_split_parent_overlap": 0, "cross_split_exact_contract_hash_overlap": 0,
        "cross_split_prompt_and_target_output_ast_overlap_ignoring_scope": 0,
        "duplicate_parent_candidates_removed": duplicate_parents_removed,
        "duplicate_message_triplets_removed": duplicate_records_removed,
        "unique_message_triplets": sum(map(len, records.values())),
        "unsupported_records": sum(row["family"].startswith("unsupported_") for rows in records.values() for row in rows),
        "split_policy": "Entire operator-composition families; all paraphrases/stages/parent variants stay together",
        "record_independence": "Four stages and up to four surface forms per parent are correlated, not independent tasks",
        "limits": "Synthetic authored conventions only; finite integer scopes; no human-intent truth, calibration, broad natural-language accuracy, or trained-model improvement claim",
        "unique_target_contracts": {split: len(values) for split, values in spec_sets.items()},
    }
    summary_path = output / "validation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    hashes[summary_path.name] = sha_file(summary_path)
    manifest = {
        "schema_version": 1, "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha_file(__file__), "parameters": {
            "parents_per_family": parents_per_family, "surfaces_per_parent": surfaces},
        "provenance": "Deterministic Python authoring with separate direct Python function oracles, Fev interpreter/typechecker and Z3 comparisons; no teacher model or external dataset",
        "label_kind": "synthetic_contract_property_not_human_intent",
        "counts": {split: len(rows) for split, rows in records.items()},
        "stages": {split: dict(Counter(row["stage"] for row in rows)) for split, rows in records.items()},
        "families": FAMILIES, "parent_ids": parents, "files_sha256": hashes,
        "kernel_files_sha256": {str(path.relative_to(ROOT)): sha_file(path) for path in (
            ROOT / "fev/ir.py", ROOT / "fev/interpreter.py", ROOT / "fev/solver_backend.py",
            ROOT / "fev/model_adapter.py", ROOT / "fev/schemas.py")},
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "split_policy": summary["split_policy"], "validation": summary,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/synthetic-v1")
    parser.add_argument("--parents-per-family", type=int, default=16)
    parser.add_argument("--surfaces", type=int, default=4)
    args = parser.parse_args()
    manifest = generate(args.output, args.parents_per_family, args.surfaces)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"],
                      "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
