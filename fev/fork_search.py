"""Finite, sequential model search; proposals never supply checking authority."""
from dataclasses import asdict
import time
import uuid

from .actions import action_matrix
from .decision import decide
from .intake import validate_claim, validate_fork_metadata
from .ir import OBSERVATION_SCHEMA, parse_and_typecheck
from .scoring import rank_forks
from .schemas import AST_GUIDE, Budgets, digest
from .solver_backend import Scope, check_consistency, compare, environment_for_scope


class ForkService:
    def __init__(self, proposer, ledger, budgets=None):
        self.proposer, self.ledger = proposer, ledger
        self.budgets = budgets or Budgets()

    def analyze(self, raw_text, scope=None, context=None):
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("request must be nonempty text")
        if len(raw_text) > 16000:
            raise ValueError("v0 request is limited to 16000 characters")
        scope = scope or Scope()
        started = time.monotonic()
        b = self.budgets
        forks, comparisons, unresolved, frontier, unsupported = [], [], [], [], []
        claims, axes, model_refs, repair_queue = [], [], [], []
        specs = {}
        usage = {"model_calls": 0, "generated_tokens": 0, "solver_ms": 0.0, "repairs": 0}
        stop = None
        base = {"raw_text": raw_text, "scope": scope.to_dict(), "language": AST_GUIDE,
                "context": context or {}, "confirmed_constraints": [],
                "instruction": "Propose observable complete contracts only for the supported integer-list domain. Preserve unsupported requirements. Source spans are Python character offsets into raw_text; explicit_text requires the exact substring. Never set user_confirmed. Assumptions include chosen error behavior and domain restrictions. Use no calibrated probability."}

        def limit():
            if usage["model_calls"] >= b.max_model_calls:
                return "model_call_budget"
            if usage["generated_tokens"] >= b.max_generated_tokens:
                return "generated_token_budget"
            if len([f for f in forks if f["status"] in {"valid", "unresolved"}]) + len(repair_queue) >= b.max_forks:
                return "fork_budget"
            if usage["solver_ms"] >= b.solver_total_ms:
                return "solver_budget"
            return None

        def solve(operation, *args):
            remaining = b.solver_total_ms - usage["solver_ms"]
            if remaining <= 0 or b.solver_query_ms == 0:
                return {"result": "unresolved", "reason": "solver_budget", "scope_hash": scope.hash}
            t = time.monotonic()
            evidence = operation(*args, scope, timeout_ms=max(1, min(b.solver_query_ms, int(remaining))))
            usage["solver_ms"] += (time.monotonic() - t) * 1000
            return evidence

        def absorb(data, stage):
            if not isinstance(data, dict):
                raise ValueError("model response must be an object")
            allowed = {
                "intake": {"claims", "ambiguity_axes", "unsupported_requirements"},
                "proposal": {"forks", "unsupported_requirements", "search_frontier"},
                "expansion": {"forks", "unsupported_requirements", "search_frontier"},
                "repair": {"forks", "unsupported_requirements", "search_frontier"},
                "critic": {"critic_notes", "unsupported_requirements", "search_frontier"},
            }[stage]
            if set(data) - allowed:
                raise ValueError(f"unsupported response fields: {sorted(set(data) - allowed)}")
            for name, items in data.items():
                if not isinstance(items, list):
                    raise ValueError(f"{name} must be a list")
            for name in ("unsupported_requirements", "search_frontier", "critic_notes"):
                if any(not isinstance(item, str) or not item.strip() for item in data.get(name, [])):
                    raise ValueError(f"{name} items must be nonempty strings")
            for claim in data.get("claims", []):
                try:
                    accepted = validate_claim(raw_text, claim)
                    if accepted not in claims:
                        claims.append(accepted)
                except (ValueError, TypeError, KeyError) as exc:
                    unresolved.append({"kind": "source_diagnostic", "stage": stage, "candidate": claim, "diagnostic": str(exc)})
            for axis in data.get("ambiguity_axes", []):
                if isinstance(axis, dict) and isinstance(axis.get("axis_id"), str):
                    if axis not in axes:
                        axes.append(axis)
                else:
                    unresolved.append({"kind": "axis_diagnostic", "candidate": axis})
            for item in data.get("unsupported_requirements", []):
                if item not in unsupported:
                    unsupported.append(item)
            for item in data.get("search_frontier", []):
                if item not in frontier:
                    frontier.append(item)
            candidates = data.get("forks", [])
            if not isinstance(candidates, list):
                raise ValueError("forks must be a list")
            if len(candidates) > (1 if stage in {"expansion", "repair"} else 2):
                raise ValueError("too many forks for this search stage")
            for candidate_index, candidate in enumerate(candidates):
                if len([f for f in forks if f["status"] in {"valid", "unresolved"}]) + len(repair_queue) >= b.max_forks:
                    frontier.extend({"kind": "unprocessed_candidate", "candidate": c} for c in candidates[candidate_index:])
                    break
                fork_id = f"F{len(forks) + 1}"
                try:
                    if not isinstance(candidate, dict):
                        raise ValueError("fork must be an object")
                    allowed_fork_fields = {"label", "assumptions", "selected_axis_options", "typed_spec_ir",
                                           "unresolved_axes", "unsupported_requirements", "parent_id"}
                    if set(candidate) - allowed_fork_fields:
                        raise ValueError(f"unsupported fork fields: {sorted(set(candidate) - allowed_fork_fields)}")
                    fork = validate_fork_metadata(raw_text, candidate)
                    spec = parse_and_typecheck(fork["typed_spec_ir"])
                except (ValueError, TypeError, KeyError) as exc:
                    diagnostic = {"kind": "compiler_diagnostic", "candidate": candidate, "diagnostic": str(exc), "stage": stage}
                    unresolved.append(diagnostic)
                    repair_queue.append(diagnostic)
                    continue
                fork.update(fork_id=fork_id, parent_id=candidate.get("parent_id"), typed_spec_ir=spec.to_dict(), spec_hash=spec.hash)
                consistency = solve(check_consistency, spec)
                fork["consistency_evidence"] = consistency
                fork["consistency_evidence_ref"] = self.ledger.put_artifact(consistency)
                fork["status"] = "valid" if consistency["result"] == "sat" else "unresolved"
                if fork["status"] == "unresolved":
                    frontier.append({"kind": "unresolved_consistency", "fork_id": fork_id,
                                     "result": consistency["result"], "reason": consistency.get("reason"),
                                     "evidence_ref": fork["consistency_evidence_ref"]})
                if fork["unsupported_requirements"] or fork["unresolved_axes"]:
                    fork["status"] = "unresolved"
                    frontier.append({"kind": "incomplete_contract", "fork_id": fork_id,
                                     "unresolved_axes": fork["unresolved_axes"], "unsupported_requirements": fork["unsupported_requirements"]})
                if fork["status"] == "valid":
                    for previous in [f for f in forks if f["status"] == "valid"]:
                        comparison = solve(compare, specs[previous["fork_id"]], spec)
                        comparison.update(left_hash=previous["spec_hash"], right_hash=spec.hash)
                        comparison["evidence_ref"] = self.ledger.put_artifact(comparison)
                        comparisons.append(comparison)
                        if comparison["result"] == "equivalent_in_scope":
                            fork["status"] = "duplicate"
                            fork["duplicate_of"] = previous["fork_id"]
                            break
                forks.append(fork)
                specs[fork_id] = spec

        def call(stage, packet, allowance):
            nonlocal stop
            stop = limit()
            if stop:
                return False
            maximum = min(allowance, b.max_generated_tokens - usage["generated_tokens"])
            usage["model_calls"] += 1
            try:
                record = self.proposer.propose(stage, packet, max_tokens=maximum)
            except Exception as exc:
                record = getattr(exc, "record", {"stage": stage, "error": str(exc), "packet": packet})
                tokens = getattr(exc, "generated_tokens", maximum)
                usage["generated_tokens"] += tokens if type(tokens) is int and tokens >= 0 else maximum
                ref = self.ledger.put_artifact(record)
                model_refs.append(ref)
                unresolved.append({"kind": "model_failure", "stage": stage, "diagnostic": str(exc), "evidence_ref": ref})
                return True
            model_refs.append(self.ledger.put_artifact(record))
            tokens = record.get("generated_tokens", maximum)
            usage["generated_tokens"] += tokens if type(tokens) is int and tokens >= 0 else maximum
            try:
                absorb(record["data"], stage)
            except (ValueError, TypeError, KeyError) as exc:
                unresolved.append({"kind": "response_diagnostic", "stage": stage, "diagnostic": str(exc), "evidence_ref": model_refs[-1]})
            return True

        def compact_summaries():
            summaries = []
            for fork in forks:
                if fork["status"] not in {"valid", "unresolved"}:
                    continue
                summary = {key: fork.get(key) for key in (
                    "fork_id", "label", "typed_spec_ir", "status", "unresolved_axes", "unsupported_requirements")}
                summary["assumptions"] = [{key: assumption[key] for key in ("text", "provenance", "start", "end")
                                           if key in assumption} for assumption in fork["assumptions"]]
                summaries.append(summary)
            return summaries

        call("intake", dict(base), 700)
        # These two packets are identical and never contain the other's proposals.
        initial_packet = {**base, "claims": list(claims), "ambiguity_axes": list(axes)}
        for _ in range(2):
            if not call("proposal", initial_packet, 1100):
                break
        for _ in range(2):
            summaries = compact_summaries()
            if not call("expansion", {**base, "claims": claims, "ambiguity_axes": axes, "retained": summaries,
                    "witnesses": [c.get("witness") for c in comparisons if c["result"] == "distinct"],
                    "instruction": "Find material observable behavior absent from retained contracts. Return at most ONE compact new fork. Return zero forks if no novelty found; this is not a completeness claim."}, 1000):
                break
        call("critic", {**base, "claims": claims, "ambiguity_axes": axes,
                        "retained": compact_summaries(), "search_frontier": frontier,
                        "unsupported_requirements": unsupported}, 500)
        # Repairs cannot renew any global budget. The old diagnostic remains audit evidence.
        for diagnostic in repair_queue[:b.max_repairs]:
            if not call("repair", {**base, "compiler_diagnostic": diagnostic}, 700):
                break
            usage["repairs"] += 1
        try:
            model_manifest = self.proposer.manifest()
        except Exception as exc:
            model_manifest = {"status": "unavailable", "error": str(exc)}
            unresolved.append({"kind": "model_manifest_unavailable", "diagnostic": str(exc)})
        stop = limit() or "bounded_schedule_complete"
        if stop != "bounded_schedule_complete":
            frontier.append({"kind": "resource_limit", "stop_reason": stop, "remaining_work": "discovery, criticism or comparison may be incomplete"})
        if unresolved:
            frontier.append({"kind": "unresolved_diagnostics", "count": len(unresolved)})
        decision, question = decide(forks, comparisons, frontier, unsupported)
        result = {"schema_version": 1, "request_id": str(uuid.uuid4()),
                  "task": {"raw_text": raw_text, "context_snapshot_hash": digest(context or {}),
                           "context": context or {}, "domain_adapter_id": "integer-list-selection-v1",
                           "scope": scope.to_dict(), "observation_schema": OBSERVATION_SCHEMA,
                           "environment": environment_for_scope(scope),
                           "claims": claims, "ambiguity_axes": axes, "model_manifest": model_manifest,
                           "budgets": asdict(b), "policy": "explicit_user_contract_selection"},
                  "forks": forks, "unresolved_candidates": unresolved, "search_frontier": frontier,
                  "unsupported_requirements": unsupported, "comparisons": comparisons,
                  "action_matrix": action_matrix(forks, comparisons), "scores": rank_forks(forks),
                  "coverage_status": "bounded_search_not_complete", "stop_reason": stop,
                  "resource_usage": {**usage, "elapsed_seconds": time.monotonic() - started},
                  "decision": decision, "question_or_contract": question,
                  "limitations": ["Natural-language translations and coverage remain model proposals.",
                                  "Equivalence and conformance apply only to the encoded IR, input scope and complete tagged observation.",
                                  "Rankings are uncalibrated; no intent probabilities or completeness guarantee.",
                                  "One checkpoint supplies all model roles; prompt diversity is not independent review."],
                  "evidence_refs": model_refs}
        result["run_hash"] = self.ledger.save_run(result)
        return result
