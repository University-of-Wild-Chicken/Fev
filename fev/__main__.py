"""Local CLI: analysis, inspection, explicit selection, and IR execution."""
import argparse
import json
from pathlib import Path
import sys

from .commitments import Ledger
from .fork_search import ForkService
from .implementation import execute
from .model_adapter import OllamaProposer, ReplayProposer
from .schemas import Budgets
from .solver_backend import Scope


def emit(value, output=None):
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def parser():
    p = argparse.ArgumentParser(description="Compile bounded semantic forks with evidence; select a contract explicitly.")
    p.add_argument("--ledger", default=".fev", help="SQLite and content-addressed evidence directory")
    commands = p.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect the installed local model and runtime")
    analyze = commands.add_parser("analyze", help="Propose and compare integer-list contracts")
    analyze.add_argument("request", nargs="?", help="Requirement text (use --request-file for exact file content)")
    analyze.add_argument("--request-file")
    analyze.add_argument("--replay", help="Recorded model response JSON array; no inference")
    analyze.add_argument("--min-value", type=int, default=-8)
    analyze.add_argument("--max-value", type=int, default=8)
    analyze.add_argument("--max-length", type=int, default=5)
    analyze.add_argument("--min-length", type=int, default=0)
    analyze.add_argument("--max-calls", type=int, default=8)
    analyze.add_argument("--max-tokens", type=int, default=6000)
    analyze.add_argument("--output")
    for command in (doctor, analyze):
        command.add_argument("--url", default="http://127.0.0.1:11435")
        command.add_argument("--model", default="aix-qwen3.8:27b-ud-q3_k_xl")
    show = commands.add_parser("show", help="Read a saved run")
    show.add_argument("request_id")
    select = commands.add_parser("select", help="Select the entire saved contract, including error cases")
    select.add_argument("request_id")
    select.add_argument("fork_id")
    select.add_argument("--reason", required=True, help="Your selection statement; recorded as user provenance")
    select.add_argument("--run-hash", help="Reject stale run selection")
    select.add_argument("--accept-limitations", action="store_true", help="Acknowledge outstanding search work; never admits an incomplete selected contract")
    run = commands.add_parser("run", help="Execute a current commitment through the reference interpreter")
    run.add_argument("commitment_id")
    run.add_argument("input", help="JSON integer list")
    export = commands.add_parser("export-replay", help="Export saved proposal responses for deterministic replay")
    export.add_argument("request_id")
    export.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate", help="Check the engineering corpus against independent Python oracles")
    evaluation.add_argument("--output")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        ledger = Ledger(args.ledger)
        if args.command == "doctor":
            emit(OllamaProposer(url=args.url, model=args.model).manifest())
        elif args.command == "analyze":
            if bool(args.request_file) == bool(args.request):
                p.error("provide either request text or --request-file")
            request = Path(args.request_file).read_text(encoding="utf-8") if args.request_file else args.request
            proposer = ReplayProposer(json.loads(Path(args.replay).read_text())) if args.replay else OllamaProposer(url=args.url, model=args.model)
            scope = Scope(min_value=args.min_value, max_value=args.max_value, max_length=args.max_length, min_length=args.min_length)
            result = ForkService(proposer, ledger, Budgets(max_model_calls=args.max_calls, max_generated_tokens=args.max_tokens)).analyze(request, scope)
            emit(result, args.output)
        elif args.command == "show":
            emit(ledger.get_run(args.request_id))
        elif args.command == "select":
            emit(ledger.commit(args.request_id, args.fork_id, "user_decision", args.reason,
                               expected_run_hash=args.run_hash, accept_limitations=args.accept_limitations))
        elif args.command == "run":
            emit(execute(ledger, args.commitment_id, json.loads(args.input)))
        elif args.command == "export-replay":
            result = ledger.get_run(args.request_id)
            emit([ledger.get_artifact(ref) for ref in result["evidence_refs"]], args.output)
        elif args.command == "evaluate":
            from .evaluation import evaluate_kernel
            result = evaluate_kernel()
            emit(result, args.output)
            return 1 if result["failures"] else 0
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
