#!/usr/bin/env python3
"""Prepare/check/run an isolated Fev QLoRA pilot using pinned local AIx code.

The imported trainer and pretrained HF checkpoint are read-only dependencies.
The generated adapter is never imported into Ollama or activated automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_training_data import sha_file, validate_record

BASE = Path("/home/augustus/AIx/.local/hf-training/models--unsloth--Qwen3.8-27B-unsloth-bnb-4bit/snapshots/8aa5f05d26b7205477066e1449e0af13f762a299")
RUNNER = Path("/home/augustus/AIx/scripts/train_midlang_offload.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_rows(records, count, seed):
    """Balance stage first and cycle families, then sample parents reproducibly."""
    families = sorted({row["family"] for row in records})
    stages = ("intake", "proposal", "repair", "critic")
    selected, used = [], set()
    pools = {}
    for stage in stages:
        for family in families:
            pool = [row for row in records if row["stage"] == stage and row["family"] == family]
            random.Random(f"{seed}:{family}:{stage}").shuffle(pool)
            pools[stage, family] = pool
    for position in range(count):
        stage = stages[position % len(stages)]
        stage_families = [family for family in families if pools[stage, family]]
        family = stage_families[position % len(stage_families)]
        pool = pools[stage, family]
        candidate = next((row for row in pool if row["id"] not in used), None)
        if candidate is None:
            candidate = next((row for row in records if row["id"] not in used), None)
        if candidate is None:
            raise ValueError("Not enough distinct source rows for pilot")
        selected.append(candidate)
        used.add(candidate["id"])
    return selected


def prepare(dataset_dir, prepared_dir, train_rows=20, validation_rows=8, seed=20261002):
    if type(train_rows) is not int or type(validation_rows) is not int or min(train_rows, validation_rows) < 1:
        raise ValueError("Training and validation row counts must be positive integers")
    dataset_dir, prepared_dir = Path(dataset_dir).resolve(), Path(prepared_dir).resolve()
    if prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise ValueError("Prepared directory is not empty; refuse to overwrite a frozen pilot")
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    selected = {}
    for split, count in (("train", train_rows), ("validation", validation_rows)):
        path = dataset_dir / (split + ".jsonl")
        if sha_file(path) != manifest["files_sha256"][path.name]:
            raise ValueError("Source dataset hash mismatch")
        rows = read_rows(path)
        if any(row["split"] != split or manifest["families"][row["family"]] != split for row in rows):
            raise ValueError("Source split/family mismatch")
        selected[split] = select_rows(rows, count, seed)
        for row in selected[split]:
            validate_record(row)
    if {r["parent_id"] for r in selected["train"]} & {r["parent_id"] for r in selected["validation"]}:
        raise ValueError("Pilot parent leakage")
    if {r["family"] for r in selected["train"]} & {r["family"] for r in selected["validation"]}:
        raise ValueError("Pilot semantic family leakage")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for split, rows in selected.items():
        path = prepared_dir / (split + ".jsonl")
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        file_hashes[path.name] = sha_file(path)
        index = [{"id": row["id"], "parent_id": row["parent_id"], "family": row["family"],
                  "prompt_sha256": hashlib.sha256(row["messages"][-2]["content"].encode()).hexdigest(),
                  "target_sha256": hashlib.sha256(row["messages"][-1]["content"].encode()).hexdigest()}
                 for row in rows]
        path = prepared_dir / (split + "-index.jsonl")
        path.write_text("".join(canonical(row) + "\n" for row in index))
        file_hashes[path.name] = sha_file(path)
    config = {
        "schema_version": 1, "purpose": f"bounded_experimental_{train_rows}_step_fev_adapter_pilot",
        "source_dataset": str(dataset_dir), "source_manifest_sha256": sha_file(dataset_dir / "manifest.json"),
        "source_split_hashes": {name: manifest["files_sha256"][name + ".jsonl"] for name in selected},
        "files_sha256": file_hashes, "selected_ids": {split: [row["id"] for row in rows] for split, rows in selected.items()},
        "counts": {split: len(rows) for split, rows in selected.items()},
        "family_counts": {split: dict(Counter(row["family"] for row in rows)) for split, rows in selected.items()},
        "stage_counts": {split: dict(Counter(row["stage"] for row in rows)) for split, rows in selected.items()},
        "base": str(BASE), "base_revision": BASE.name, "base_config_sha256": sha_file(BASE / "config.json"),
        "base_weights_bytes": (BASE / "model.safetensors").stat().st_size,
        "base_weights_blob": (BASE / "model.safetensors").resolve().name,
        "base_identity_note": "HF content-addressed cache filename and config hash; large checkpoint not rehashed by prepare",
        "checkpoint_format": "Hugging Face safetensors NF4, qwen3_5 architecture; not Ollama GGUF",
        "runner": str(RUNNER), "runner_sha256": sha_file(RUNNER),
        "checkpoint_helper_sha256": sha_file(RUNNER.with_name("distillation_checkpoint.py")),
        "wrapper_sha256": sha_file(__file__), "seed": seed,
        "train_config": {"epochs": 1, "probe": train_rows, "last_layers": 4, "resident_frozen_layers": 4,
                         "rank": 8, "learning_rate": 0.0002, "max_length": 4096, "loss_chunk": 32,
                         "checkpoint_every": 5, "min_host_available_gib": 2.5},
        "loss": "assistant response tokens only, exact chunked cross entropy",
        "selection": "compare held-out semantic-family response loss before/after; baseline may win",
        "no_claims": "Not calibrated human-intent probabilities or established general capability improvement",
        "test_records_loaded": False, "downloads": False, "automatic_activation": False,
    }
    (prepared_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def check(config_path):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text())
    prepared = path.parent
    if config.get("schema_version") != 1:
        raise ValueError("Unknown recipe version")
    if sha_file(__file__) != config["wrapper_sha256"]:
        raise ValueError("Training wrapper changed after preparation")
    runner = Path(config["runner"])
    if sha_file(runner) != config["runner_sha256"] or sha_file(runner.with_name("distillation_checkpoint.py")) != config["checkpoint_helper_sha256"]:
        raise ValueError("Read-only training dependency changed")
    base = Path(config["base"])
    model_config = json.loads((base / "config.json").read_text())
    if model_config.get("model_type") != "qwen3_5" or base.suffix.lower() == ".gguf":
        raise ValueError("Expected the trainable qwen3_5 Hugging Face checkpoint, not GGUF")
    if sha_file(base / "config.json") != config["base_config_sha256"]:
        raise ValueError("Base checkpoint configuration changed")
    weights = base / "model.safetensors"
    if weights.stat().st_size != config["base_weights_bytes"] or weights.resolve().name != config["base_weights_blob"]:
        raise ValueError("Base weights identity changed")
    source = Path(config["source_dataset"])
    if sha_file(source / "manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Source dataset manifest changed")
    # Deliberately never open test.jsonl during preparation/check/training.
    for split, expected in config["source_split_hashes"].items():
        if split not in ("train", "validation") or sha_file(source / (split + ".jsonl")) != expected:
            raise ValueError("Source training/validation split changed")
    rows = {}
    for name, expected in config["files_sha256"].items():
        if Path(name).name != name or sha_file(prepared / name) != expected:
            raise ValueError("Frozen pilot file changed: " + name)
    for split in ("train", "validation"):
        rows[split] = read_rows(prepared / (split + ".jsonl"))
        if [row["id"] for row in rows[split]] != config["selected_ids"][split]:
            raise ValueError("Selected row identity mismatch")
        for row in rows[split]:
            if row["split"] != split:
                raise ValueError("Unexpected split in prepared pilot")
            validate_record(row)
    if {row["family"] for row in rows["train"]} & {row["family"] for row in rows["validation"]}:
        raise ValueError("Training/validation family contamination")
    return prepared, config


def load_runner(config):
    runner = Path(config["runner"])
    sys.path.insert(0, str(runner.parent))
    spec = importlib.util.spec_from_file_location("fev_read_only_aix_training_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input_loader(prepared, config):
    def load_input(path, *, full_run=False):
        path = Path(path).resolve()
        if path.parent != prepared or path.name not in ("train.jsonl", "validation.jsonl"):
            raise ValueError("Trainer may open only frozen pilot train/validation files")
        split = path.stem
        captured = path.read_bytes()
        captured_hash = hashlib.sha256(captured).hexdigest()
        if captured_hash != config["files_sha256"][path.name]:
            raise ValueError("Training data changed")
        index_path = path.with_name(split + "-index.jsonl")
        index_bytes = index_path.read_bytes()
        index_hash = hashlib.sha256(index_bytes).hexdigest()
        if index_hash != config["files_sha256"][index_path.name]:
            raise ValueError("Training index changed")
        records = [json.loads(line) for line in captured.decode().splitlines() if line.strip()]
        ids = [row["id"] for row in records]
        if ids != config["selected_ids"][split]:
            raise ValueError("Selected example IDs changed")
        return {"records": records, "ids": ids, "index_path": index_path, "lineage": {
            "dataset_sha256": captured_hash, "train_index_sha256": index_hash,
            "source_manifest_sha256": config["source_manifest_sha256"]}}
    return load_input


def tokenize_check(prepared, config):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    module = load_runner(config)
    loader = make_input_loader(prepared, config)
    tokenizer = AutoTokenizer.from_pretrained(config["base"], local_files_only=True)
    counts = {}
    for split in ("train", "validation"):
        path = prepared / (split + ".jsonl")
        examples = module.response_examples(path, tokenizer, input_data=loader(path))
        lengths = [len(example["ids"]) for example in examples]
        if max(lengths) > config["train_config"]["max_length"]:
            raise ValueError("Pilot exceeds max_length; no silent truncation")
        counts[split] = {"rows": len(examples), "min_tokens": min(lengths), "max_tokens": max(lengths),
                         "response_tokens": sum(len(example["ids"]) - example["response_start"] for example in examples)}
    return counts


def train(config_path, output):
    prepared, config = check(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Training output already exists; preserve prior adapters/checkpoints")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true")
    module = load_runner(config)
    module.load_training_input = make_input_loader(prepared, config)
    args = SimpleNamespace(base=Path(config["base"]), dataset=prepared / "train.jsonl",
                           validation=prepared / "validation.jsonl", output=output,
                           seed=config["seed"], resume=None, **config["train_config"])
    # This wrapper changes only input validation, not optimization or kernel math.
    pilot_manifest = {"config_sha256": sha_file(config_path), "wrapper_sha256": sha_file(__file__),
                      "config": config, "output": str(output), "started_unix": time.time(),
                      "status": "running", "automatic_activation": False}
    try:
        module.run(args)
        pilot_manifest["status"] = "complete"
    except Exception as exc:
        pilot_manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if output.exists():
            (output / "failure.json").write_text(json.dumps(pilot_manifest, indent=2) + "\n")
            runner_manifest = output / "manifest.json"
            if runner_manifest.exists():
                data = json.loads(runner_manifest.read_text())
                data["status"] = "failed"
                runner_manifest.write_text(json.dumps(data, indent=2) + "\n")
        raise
    finally:
        if output.exists():
            pilot_manifest["finished_unix"] = time.time()
            (output / "fev-pilot.json").write_text(json.dumps(pilot_manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare", action="store_true")
    operation.add_argument("--check", action="store_true")
    operation.add_argument("--tokenize-check", action="store_true")
    operation.add_argument("--train", action="store_true")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/synthetic-v1")
    parser.add_argument("--prepared", type=Path, default=ROOT / "training/pilot-v1")
    parser.add_argument("--config", type=Path, default=ROOT / "training/pilot-v1/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/training/fev-pilot-v1")
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.dataset, args.prepared)
        print(json.dumps({"prepared": str(args.prepared), "counts": result["counts"]}))
    elif args.train:
        train(args.config, args.output)
    else:
        prepared, config = check(args.config)
        result = tokenize_check(prepared, config) if args.tokenize_check else {"status": "checked", "counts": config["counts"], "weights_loaded": False, "downloads": False}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
