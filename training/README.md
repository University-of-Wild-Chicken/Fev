# Fev synthetic data and local adapter pilot

The frozen `datasets/synthetic-v1` corpus contains **2,372 unique message triples**:
1,284 training, 544 validation and 544 test examples. They come from 192 authored
parent tasks: 144 supported list-function tasks and 48 unsupported-domain tasks.
Multiple stages and surface forms of a parent are correlated examples, not
independent observations. Generation removed 48 duplicate parent candidates and
28 duplicate message triples.

The targets use the production Fev JSON response schemas for intake, proposal,
syntax repair and criticism. The corpus includes explicit source spans, cases
with a source-resolved choice, scoped duplicate-contract labels, material
alternatives with checked witnesses, and 96 unsupported-domain examples. Error
conventions needed by the synthetic contracts are stated in the source.

Every target is schema checked. Every target specification is type checked and
compared against a separately written direct Python oracle for every input in its
small declared domain. The frozen validation report records **419,364 oracle
evaluations**, **144 material comparisons** and **144 equivalent comparisons**.
Every solver witness is replayed through those independent Python oracles.
These are formal properties of synthetic authored contracts; they are **not**
human-intent labels, calibration data or evidence that a trained model improves.

Entire composition families are held out. Validation covers guarded selection,
extremum thresholds, state updates and filesystem effects. Test covers compound
predicates, conditional list projections, time and randomness. Primitives are
shared, but authored parent IDs, target contract hashes and prompt/target output
ASTs ignoring scope have zero cross-split overlap. No semantic family is split by
random paraphrase. The generic schema-guide example is shared prompt context and
is excluded from the authored-contract overlap check.

Regenerate into a new directory; the generator refuses to overwrite a frozen set:

```bash
.venv/bin/python scripts/generate_training_data.py --output datasets/synthetic-v2
.venv/bin/python -m unittest discover -s tests -p test_training_data.py -v
```

Each JSONL row includes `messages`, stable IDs, family, stage, split and label
provenance. `evidence.jsonl` stores independently replayed witnesses and solver
queries. `manifest.json` binds every data file, generator and kernel source hash;
`validation.json` gives counts and limitations. Training tools parse only the
training and validation splits, never test records.

## Local pilot

The live Ollama GGUF is an inference artifact. Training uses the already cached
Hugging Face NF4 safetensors checkpoint at:

```
/home/augustus/AIx/.local/hf-training/models--unsloth--Qwen3.8-27B-unsloth-bnb-4bit/snapshots/8aa5f05d26b7205477066e1449e0af13f762a299
```

Its configuration declares `Qwen3_5ForConditionalGeneration` / `qwen3_5`.
The cache identity is the existing content-addressed weights filename and pinned
configuration hash; preparation does not rehash the 22.34 GB weights file.

The ordinary [Unsloth Qwen3.8 recipe](https://unsloth.ai/docs/models/qwen3.8/train)
documents a 24 GB QLoRA GPU target. This machine has a 16 GB RTX 5080. The pilot
instead reuses AIx's existing streamed-prefix trainer as a hash-pinned, read-only
dependency. Frozen prefix layers are stored on CPU and executed on GPU; only
rank-8 LoRA weights in the last four MLP layers train. Four frozen layers remain
resident; response-only cross entropy is evaluated in 32-token chunks.

An earlier AIx run using this approach completed 300 steps with median update
time 8.17 seconds, peak reserved VRAM about 12.79 GiB and peak host RSS about
23.50 GiB. Those are prior AIx measurements, not a Fev performance guarantee.

`training/pilot-v1/config.json` freezes 20 stratified training examples and eight
validation-family examples, their source hashes, the wrapper, dependency code and
base checkpoint identity. Tokenization was checked locally: training sequences
823–1,372 tokens and validation sequences 945–1,518 tokens, below the 4,096-token
limit. No sequence is silently truncated. The wrapper changes the original
trainer's input validation only; it does not alter its gradient mathematics.

```bash
.venv/bin/python scripts/train_lora.py --check
/home/augustus/AIx/.venv-finetune/bin/python scripts/train_lora.py --tokenize-check
/home/augustus/AIx/.venv-finetune/bin/python scripts/train_lora.py --train \
  --config training/pilot-v1/config.json \
  --output .local/training/fev-pilot-v2
```

The resident inference model must be released before training to make GPU memory
available. The wrapper never changes the AIx source/checkpoint, downloads a model,
stops a service, imports an adapter into Ollama or activates it. Existing output
directories are protected. Before/after response loss is measured on the same
eight held-out-family examples. The imported trainer retains the baseline as a
candidate and refuses to export an adapter if held-out loss does not improve.
Checkpoints and failure diagnostics remain available if the pilot fails.

`fev-pilot.json` binds the wrapper and prepared configuration to the actual run;
the imported trainer also saves loading diagnostics, gradient checks, per-step
loss/memory, validation history, checkpoint lineage and adapter-reload checks.
Even a successful 20-step pilot and lower validation loss do not establish intent
accuracy or justify replacing the resident base model. A separate held-out
behavioral evaluation and explicit deployment step are still required.

## Completed pilot

The first Fev pilot completed all 20 steps and saved its adapter to
`.local/training/fev-pilot-v1/adapter/`. Token-weighted response loss on the eight
held-out validation examples decreased from **0.5370 to 0.4080**. The trainer
reported 173.95 seconds, peak reserved CUDA memory of 6.09 GiB, and peak host RSS
of 19.33 GiB. Saving and reloading the adapter reproduced its parameters exactly
(maximum absolute reload error: zero).

The full records are `.local/training/fev-pilot-v1/manifest.json`,
`validation.json`, `steps.jsonl`, and `fev-pilot.json`. This is a small feasibility
pilot with synthetic response-loss evidence, not a completed behavioral accuracy
evaluation. The adapter was not activated; the original Ollama base model was
reloaded after training. Use a fresh output directory for another run.
