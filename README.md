# Fev

Fev is a local semantic-fork compiler. It turns a short requirement into bounded integer-list contracts, mechanically compares their observable behavior, asks a concrete material question, and preserves a selected contract for execution.

This first release implements the proposal's semantic kernel, fork service, decision service, and reference-IR execution adapter. The local Qwen model proposes meanings; Python and Z3 determine the evidence labels. Natural-language completeness and intended meaning remain unresolved by those checks.

## Run locally

The environment in this checkout is installed and ready:

```bash
cd /home/augustus/Fev
.venv/bin/fev doctor
.venv/bin/fev analyze "Return the second-largest number."
```

The default model is the installed base tag `aix-qwen3.8:27b-ud-q3_k_xl` at `http://127.0.0.1:11435`. Fev uses the existing AIx Ollama service. It does not require an API key or cloud inference. `--url` and `--model` select another installed loopback model. `doctor` records the runtime version, installed tag digest, reported model blob, quantization metadata, tokenizer metadata, template, and generation options. Reported weight identity is distinguished from independently hashing the weight file.

For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

To run without inference, use the explicitly synthetic fixture:

```bash
.venv/bin/fev analyze "Return the second-largest number." \
  --replay examples/second_largest.replay.json --output result.json
```

The JSON result contains the exact source, source-linked claims, ambiguity axes, typed contracts, checked witnesses, action matrix, uncalibrated ranks, pending work, resource usage, and the next decision. `request_id` identifies the saved run. Evidence is stored in `.fev/ledger.sqlite3` and `.fev/artifacts/`; `--ledger DIRECTORY` changes that location.

After reading a full contract, select it using the actual user's decision:

```bash
.venv/bin/fev show REQUEST_ID
.venv/bin/fev select REQUEST_ID F1 --run-hash RUN_HASH \
  --reason "I choose this full contract, including its duplicate and error behavior."
.venv/bin/fev run COMMITMENT_ID '[5,5,3]'
```

Identifiers above are placeholders from your saved result. A witness answers only the behavior shown; selecting a contract also selects its other assumptions and error cases. When search remains unresolved, selection requires `--accept-limitations`. That flag cannot admit an incomplete, unsupported, inconsistent, duplicate, or unchecked selected contract. The CLI requires explicit selection even when the retained contracts agree. It does not infer confirmation from model text or rankings.

Selecting a replacement creates a new immutable commitment and makes earlier execution evidence stale. New proposal runs alone do not replace an existing user decision. Executions run the committed IR through the reference interpreter and preserve the exact input/result and commitment bindings; they do not generate arbitrary code.

## Supported semantics

The v1 adapter supports pure lists of bounded integers, integer/boolean/list constants, sorting, stable duplicate removal, constant-index selection, length, comparisons, boolean operators, conditionals, and tagged errors. JSON is transport for a typed AST. No model-emitted Python, SQL, or SMT is executed.

Default scope is integer values `[-8,8]` and lengths `0..5`. Use `--min-value`, `--max-value`, `--min-length`, and `--max-length` to change it. Absolute kernel limits are 16 list elements, 128 AST nodes, depth 16, and integer magnitude at most `2^31-1`. Arithmetic and overflow operators are intentionally absent. Sorting and selection never mutate the input.

Errors are observable. A missing index returns `index_out_of_bounds` unless a contract chooses a different tagged error. A contract's own input restriction is part of its behavior, not a hidden shared solver assumption. Execution refuses well-formed values outside the committed checking scope.

Record updates, access control, floating point, strings, arithmetic, concurrency, and temporal behavior are unsupported. Their obligations stay visible and cannot be silently promoted to successful checks.

## Evidence and budgets

The interpreter is the reference semantics. The Z3 compiler represents symbolic list length, active elements, outputs, and errors. Every comparison first establishes that the shared environment is satisfiable. SAT difference queries produce witnesses replayed through the interpreter; UNSAT means equivalent only under the recorded scope, environment, observation schema, kernel source hashes, and solver version. Timeout, missing solver, inconsistent environment, or replay mismatch remain unresolved.

The default controller allows six retained/pending forks, eight model calls, 6,000 generated tokens, two repairs, a two-second solver-query budget, and 30 seconds of solver work. Intake, two isolated proposal calls, up to two expansions, and a critic precede bounded repairs. No-novelty and resource limits are search stops, never completeness claims. Requests are serialized against the resident model and checked for context capacity; oversize prompts remain explicit failures.

Model roles use one checkpoint and are not independent reviewers. Plausibility ranks are transparent uncalibrated heuristics. Probabilities are `null`. Conformance cells refer to exact IR actions and are not normalized into probabilities. There is no learned user-intent scorer or automatic utility/default policy in this release.

Raw proposals and API timing records are retained for replay:

```bash
.venv/bin/fev export-replay REQUEST_ID --output proposals.json
.venv/bin/fev analyze "THE SAME EXACT REQUEST" --replay proposals.json
```

Recorded model calls are bound to the request, context, language, scope, stage, and token limits. Replay rechecks the proposals with the current kernel; changed derived solver feedback is labeled. Python callers can request strict packet matching for exact-prompt audits. Regenerating GPU output from a seed is not deterministic replay.

## Verification and training data

```bash
.venv/bin/python -m pytest -q
.venv/bin/fev evaluate
```

The kernel tests exhaustively compare interpreter and Z3 outputs on tiny domains, check behavior-class comparisons, and exercise error propagation, vacuous environments, unavailable solvers, budget exhaustion, provenance rejection, evidence tampering, and commitment supersession. The 120-case engineering corpus includes 108 supported contracts and 12 unsupported requests. Its independent Python oracles check 21,456 bounded inputs. It is authored engineering data, not human-intent ground truth.

[Synthetic training data and fine-tuning instructions](training/README.md) describe the separate generated corpus, its mechanical validation, family splits, and model-training constraints. Synthetic targets teach IR construction and repair; they do not calibrate intent probabilities.

The first local LoRA pilot completed 20 steps on that corpus, with validation response loss decreasing from 0.5370 to 0.4080 across eight held-out examples. Its saved adapter remains separate from the resident base model. The complete test suite currently has 121 passing tests.

The original [construction proposal](docs/construction-proposal-v0.1.txt) remains the roadmap. Later work includes additional domain adapters, a UI, human intent-label collection and calibrated scoring, code-generation handoff, and a controlled three-baseline model evaluation. This release makes no model-accuracy or population-level failure-rate claim.

## Layout

- `fev/ir.py`, `interpreter.py`, `solver_backend.py`: trusted typed kernel and scoped obligations.
- `fev/model_adapter.py`: local structured proposals, pinned manifests, replay.
- `fev/intake.py`, `fork_search.py`, `actions.py`, `decision.py`, `scoring.py`: provenance, finite search, consequences, policy.
- `fev/commitments.py`, `implementation.py`: immutable evidence and committed IR execution.
- `fev/evaluation.py`, `tests/`, `examples/`: independently checked engineering fixtures and regressions.

Runtime API references: [Ollama chat](https://docs.ollama.com/api/chat), [installed-model tags](https://docs.ollama.com/api/tags). Solver result semantics: [Z3 basic commands](https://microsoft.github.io/z3guide/docs/logic/basiccommands/).
