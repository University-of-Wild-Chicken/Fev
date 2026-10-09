"""Immutable, content-addressed run, commitment, and evidence storage.

SQLite serializes decisions and stores append-only indices. JSON artifacts are
canonical, atomically published, and verified on every read. Invalidating old
evidence means retaining its history while refusing to treat it as current.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterator
import uuid


class LedgerError(ValueError):
    """A ledger operation cannot establish the requested claim."""


class IntegrityError(LedgerError):
    """Stored content is missing or does not match its digest/index."""


class CommitmentError(LedgerError):
    """The proposed selection cannot become an executable commitment."""


class StaleCommitmentError(CommitmentError):
    """A newer run or commitment supersedes the referenced decision."""


def _canonical(data: Any) -> bytes:
    def validate(value: Any) -> None:
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise LedgerError("JSON object keys must be strings")
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise LedgerError("Artifacts must contain only JSON values")

    validate(data)
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LedgerError(f"Invalid JSON artifact: {exc}") from exc


def _digest(data: Any) -> str:
    return hashlib.sha256(_canonical(data)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    digest TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    artifact_hash TEXT NOT NULL REFERENCES artifacts(digest),
    UNIQUE(request_id, revision)
);
CREATE TABLE IF NOT EXISTS commitments (
    commitment_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    artifact_hash TEXT NOT NULL REFERENCES artifacts(digest),
    UNIQUE(request_id, version)
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    commitment_id TEXT NOT NULL REFERENCES commitments(commitment_id),
    artifact_hash TEXT NOT NULL REFERENCES artifacts(digest)
);
CREATE INDEX IF NOT EXISTS evidence_by_commitment ON evidence(commitment_id);
"""


class Ledger:
    """A ledger directory containing ``ledger.sqlite3`` and ``artifacts``.

    ``commit`` is the trusted boundary for an actual user selection. Proposal
    callers must never pass ``user_decision`` on the model's behalf. Explicit
    policy selection is deliberately unsupported in the first release.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "ledger.sqlite3"
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            for table in ("artifacts", "runs", "commitments", "evidence"):
                for operation in ("UPDATE", "DELETE"):
                    connection.execute(
                        f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation.lower()} "
                        f"BEFORE {operation} ON {table} BEGIN "
                        "SELECT RAISE(ABORT, 'ledger rows are immutable'); END"
                    )

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        if write:
            connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _artifact_path(self, digest: str) -> Path:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise IntegrityError("Artifact reference must be a lowercase SHA-256 digest")
        return self.artifacts_dir / f"{digest}.json"

    def _put_artifact(self, data: Any, connection: sqlite3.Connection) -> str:
        payload = _canonical(data)
        digest = hashlib.sha256(payload).hexdigest()
        destination = self._artifact_path(digest)
        if not destination.exists():
            descriptor, temporary = tempfile.mkstemp(prefix=".artifact-", dir=self.artifacts_dir)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    pass
                directory_fd = os.open(self.artifacts_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                os.unlink(temporary)
        # Existing files are never overwritten, including corrupted files.
        self.get_artifact(digest)
        connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?)", (digest, _now()))
        return digest

    def put_artifact(self, data: Any) -> str:
        with self._connection(write=True) as connection:
            return self._put_artifact(data, connection)

    def get_artifact(self, digest: str) -> Any:
        path = self._artifact_path(digest)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"Missing or unreadable artifact {digest}") from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise IntegrityError(f"Artifact digest mismatch: {digest}")
        try:
            data = json.loads(payload)
            if _canonical(data) != payload:
                raise IntegrityError(f"Artifact is not canonical JSON: {digest}")
        except (ValueError, UnicodeError) as exc:
            raise IntegrityError(f"Invalid JSON artifact {digest}") from exc
        return data

    @staticmethod
    def _latest_run(connection: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM runs WHERE request_id=? ORDER BY revision DESC LIMIT 1", (request_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"Unknown request: {request_id}")
        return row

    def _read_run(self, row: sqlite3.Row) -> dict[str, Any]:
        data = self.get_artifact(row["artifact_hash"])
        if not isinstance(data, dict) or data.get("request_id") != row["request_id"]:
            raise IntegrityError("Run index does not match its artifact")
        self._verify_references(data)
        return data

    def _verify_references(self, data: Any) -> None:
        """Verify explicit evidence references without interpreting arbitrary text."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ("evidence_ref", "consistency_evidence_ref", "witness_or_query_ref") and value is not None:
                    self.get_artifact(value)
                elif key == "evidence_refs":
                    if not isinstance(value, list):
                        raise IntegrityError("evidence_refs must be an array of artifact digests")
                    for reference in value:
                        self.get_artifact(reference)
                else:
                    self._verify_references(value)
        elif isinstance(data, list):
            for item in data:
                self._verify_references(item)

    def save_run(self, result: dict[str, Any]) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("request_id"), str) or not result["request_id"].strip():
            raise LedgerError("A run needs a nonempty request_id")
        snapshot = dict(result)
        snapshot.pop("run_hash", None)
        snapshot.pop("run_revision", None)
        self._verify_references(snapshot)
        with self._connection(write=True) as connection:
            digest = self._put_artifact(snapshot, connection)
            revision = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 FROM runs WHERE request_id=?", (result["request_id"],)
            ).fetchone()[0]
            connection.execute("INSERT INTO runs(request_id, revision, artifact_hash) VALUES (?, ?, ?)",
                               (result["request_id"], revision, digest))
        return digest

    def get_run(self, request_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = self._latest_run(connection, request_id)
            result = self._read_run(row)
        return {**result, "run_hash": row["artifact_hash"], "run_revision": row["revision"]}

    def run_hash(self, request_id: str) -> str:
        return self.get_run(request_id)["run_hash"]

    @staticmethod
    def _selection(run: dict[str, Any], fork_id: str, accept_limitations: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        forks = run.get("forks")
        if not isinstance(forks, list):
            raise CommitmentError("Run has no fork registry")
        matches = [fork for fork in forks if isinstance(fork, dict) and fork.get("fork_id") == fork_id]
        if len(matches) != 1:
            raise CommitmentError("Selected fork must exist exactly once")
        fork = matches[0]
        if fork.get("status") != "valid":
            raise CommitmentError("Only a valid, nonduplicate fork can be committed")
        if fork.get("unsupported_requirements") or fork.get("unresolved_axes"):
            raise CommitmentError("Selected fork has unsupported requirements or unresolved axes")
        consistency = fork.get("consistency_evidence", {})
        if not isinstance(consistency, dict) or consistency.get("result") != "sat":
            raise CommitmentError("Selected fork requires satisfiable consistency evidence")
        if consistency.get("status") != "consistent_in_scope" or consistency.get("witness_replayed") is not True:
            raise CommitmentError("Consistency evidence requires a replayed in-scope witness")
        specification = fork.get("typed_spec_ir")
        if not isinstance(specification, dict) or not specification:
            raise CommitmentError("Selected fork needs typed_spec_ir")
        if fork.get("spec_hash") != _digest(specification):
            raise IntegrityError("Selected specification hash does not match typed_spec_ir")
        from .ir import parse_and_typecheck
        if parse_and_typecheck(specification).hash != fork["spec_hash"]:
            raise IntegrityError("Selected specification is not canonical typed IR")
        task = run.get("task")
        if not isinstance(task, dict) or any(key not in task for key in ("scope", "environment", "observation_schema")):
            raise CommitmentError("Task must preserve scope, environment, and observation_schema")
        if not isinstance(task["scope"], dict):
            raise CommitmentError("Task scope must be an object")
        if consistency.get("spec_hashes") != [fork["spec_hash"]]:
            raise IntegrityError("Consistency evidence must reference exactly the selected specification")
        for digest_field, value in (("scope_hash", task["scope"]),
                                    ("environment_hash", task["environment"]),
                                    ("observation_hash", task["observation_schema"])):
            if consistency.get(digest_field) != _digest(value):
                raise IntegrityError(f"Consistency evidence has a different {digest_field}")
        for field in ("scope", "environment", "observation_schema"):
            if consistency.get(field) != task[field]:
                raise IntegrityError(f"Consistency evidence describes a different {field}")
        runtime = consistency.get("runtime_manifest")
        if not isinstance(runtime, dict) or not runtime or consistency.get("runtime_hash") != _digest(runtime):
            raise IntegrityError("Consistency evidence requires its exact runtime manifest and hash")
        unresolved = (run.get("search_frontier") or run.get("unresolved_candidates") or
                      run.get("unsupported_requirements") or
                      any(isinstance(item, dict) and item.get("result") == "unresolved"
                          for item in run.get("comparisons", [])))
        if unresolved and not accept_limitations:
            raise CommitmentError("Unresolved search limitations require explicit user acknowledgment")
        return fork, task

    @staticmethod
    def _error_cases(specification: dict[str, Any]) -> list[dict[str, Any]]:
        errors = [
            {"code": "invalid_input", "condition": "input is not an integer list; booleans are rejected"},
            {"code": "input_out_of_bounds", "condition": "input violates the frozen input signature",
             "input_signature": specification["input"]},
        ]

        def visit(node: Any, path: str) -> None:
            if isinstance(node, dict):
                if node.get("op") == "error":
                    errors.append({"code": node["code"], "condition": "error expression is evaluated", "expression_path": path})
                elif node.get("op") == "at":
                    errors.append({"code": node["error"], "condition": "selected index is absent",
                                   "index": node["index"], "expression_path": path})
                for key, child in node.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    visit(child, f"{path}[{index}]")

        visit(specification["output"], "$.output")
        return errors

    def _verify_confirmation(self, value: Any, connection: sqlite3.Connection) -> None:
        """Only a ledger decision can support inherited user-confirmed labels."""
        if isinstance(value, dict):
            if value.get("provenance") == "user_confirmed":
                source_id = value.get("commitment_id")
                if not isinstance(source_id, str):
                    raise CommitmentError("user_confirmed provenance requires an actual prior commitment_id")
                _, prior = self._commitment(connection, source_id)
                if prior.get("selection_source") != "user_decision":
                    raise CommitmentError("user_confirmed provenance must reference an actual user decision")
                statement = {key: item for key, item in value.items() if key not in ("provenance", "commitment_id")}
                confirmed = [{key: item for key, item in assumption.items() if key not in ("provenance", "commitment_id")}
                             for assumption in prior.get("assumptions", []) if isinstance(assumption, dict)]
                if statement not in confirmed:
                    raise CommitmentError("A prior user decision does not confirm this assumption")
            for child in value.values():
                self._verify_confirmation(child, connection)
        elif isinstance(value, list):
            for child in value:
                self._verify_confirmation(child, connection)

    def commit(self, request_id: str, fork_id: str, selection_source: str,
               selection_text: str, expected_run_hash: str | None = None,
               *, accept_limitations: bool = False) -> dict[str, Any]:
        if selection_source != "user_decision":
            raise CommitmentError("Only an actual user_decision may select a contract in this release")
        if not isinstance(selection_text, str) or not selection_text.strip():
            raise CommitmentError("The actual user's selection text must be preserved")
        if type(accept_limitations) is not bool:
            raise CommitmentError("accept_limitations must be an explicit boolean")
        with self._connection(write=True) as connection:
            row = self._latest_run(connection, request_id)
            if expected_run_hash is not None and row["artifact_hash"] != expected_run_hash:
                raise StaleCommitmentError("The request changed after the selection was reviewed")
            run = self._read_run(row)
            fork, task = self._selection(run, fork_id, accept_limitations)
            reference = fork.get("consistency_evidence_ref")
            if not reference or self.get_artifact(reference) != fork["consistency_evidence"]:
                raise IntegrityError("Consistency evidence must match its stored artifact reference")
            self._verify_confirmation(fork, connection)
            self._verify_confirmation(task.get("claims", []), connection)
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM commitments WHERE request_id=?", (request_id,)
            ).fetchone()[0]
            commitment_id = str(uuid.uuid4())
            record: dict[str, Any] = {
                "commitment_id": commitment_id,
                "request_id": request_id,
                "version": version,
                "created_at": _now(),
                "fork_id": fork_id,
                "spec_hash": fork["spec_hash"],
                "typed_spec_ir": fork["typed_spec_ir"],
                "selected_fork": fork,
                "run_hash": row["artifact_hash"],
                "run_revision": row["revision"],
                "selection_source": selection_source,
                "selection_text": selection_text,
                "selection_provenance": {
                    "source": selection_source,
                    "text": selection_text,
                    "provenance": "user_confirmed",
                    "accepted_limitations": accept_limitations,
                },
                "assumptions": fork.get("assumptions", []),
                "error_cases": self._error_cases(fork["typed_spec_ir"]),
                "limitations": run.get("limitations", []),
                "unresolved_candidates": run.get("unresolved_candidates", []),
                "search_frontier": run.get("search_frontier", []),
                "unsupported_requirements": run.get("unsupported_requirements", []),
                "decision": run.get("decision"),
            }
            for field in ("scope", "environment", "observation_schema"):
                record[field] = task[field]
                record[f"{field}_hash"] = self._put_artifact(task[field], connection)
            self._put_artifact(fork["typed_spec_ir"], connection)
            digest = self._put_artifact(record, connection)
            connection.execute("INSERT INTO commitments VALUES (?, ?, ?, ?, ?)",
                               (commitment_id, request_id, version, row["run_id"], digest))
        return record

    def _commitment(self, connection: sqlite3.Connection, id_or_request: str) -> tuple[sqlite3.Row, dict[str, Any]]:
        row = connection.execute("SELECT * FROM commitments WHERE commitment_id=?", (id_or_request,)).fetchone()
        if row is None:
            row = connection.execute(
                "SELECT * FROM commitments WHERE request_id=? ORDER BY version DESC LIMIT 1", (id_or_request,)
            ).fetchone()
        if row is None:
            raise LedgerError(f"Unknown commitment or uncommitted request: {id_or_request}")
        record = self.get_artifact(row["artifact_hash"])
        if not isinstance(record, dict) or any(record.get(key) != row[key] for key in ("commitment_id", "request_id", "version")):
            raise IntegrityError("Commitment index does not match its artifact")
        run_row = connection.execute("SELECT * FROM runs WHERE run_id=?", (row["run_id"],)).fetchone()
        if run_row is None or run_row["artifact_hash"] != record.get("run_hash"):
            raise IntegrityError("Commitment run reference does not match its artifact")
        self._read_run(run_row)
        for field, digest_field in (("typed_spec_ir", "spec_hash"), ("scope", "scope_hash"),
                                    ("environment", "environment_hash"), ("observation_schema", "observation_schema_hash")):
            if self.get_artifact(record[digest_field]) != record[field]:
                raise IntegrityError(f"Commitment {field} differs from its referenced artifact")
        return row, record

    def get_commitment(self, id_or_request: str) -> dict[str, Any]:
        with self._connection() as connection:
            return self._commitment(connection, id_or_request)[1]

    @staticmethod
    def _is_current(connection: sqlite3.Connection, row: sqlite3.Row) -> bool:
        latest = connection.execute(
            "SELECT commitment_id FROM commitments WHERE request_id=? ORDER BY version DESC LIMIT 1", (row["request_id"],)
        ).fetchone()
        return latest is not None and latest["commitment_id"] == row["commitment_id"]

    def assert_current(self, commitment_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row, record = self._commitment(connection, commitment_id)
            if not self._is_current(connection, row):
                raise StaleCommitmentError("A newer contract version invalidates this commitment")
            return record

    def save_evidence(self, commitment_id: str, data: Any) -> dict[str, Any]:
        with self._connection(write=True) as connection:
            row, commitment = self._commitment(connection, commitment_id)
            if not self._is_current(connection, row):
                raise StaleCommitmentError("Cannot attach current evidence to a superseded commitment")
            data_hash = self._put_artifact(data, connection)
            record = {"evidence_id": str(uuid.uuid4()), "commitment_id": commitment["commitment_id"],
                      "created_at": _now(), "data_hash": data_hash}
            digest = self._put_artifact(record, connection)
            connection.execute("INSERT INTO evidence VALUES (?, ?, ?)",
                               (record["evidence_id"], record["commitment_id"], digest))
        return {**record, "valid": True, "current": True}

    def list_evidence(self, commitment_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            commitment_row, commitment = self._commitment(connection, commitment_id)
            current = self._is_current(connection, commitment_row)
            rows = connection.execute("SELECT * FROM evidence WHERE commitment_id=? ORDER BY rowid",
                                      (commitment["commitment_id"],)).fetchall()
            results = []
            for row in rows:
                record = self.get_artifact(row["artifact_hash"])
                if not isinstance(record, dict) or any(record.get(key) != row[key] for key in ("evidence_id", "commitment_id")):
                    raise IntegrityError("Evidence index does not match its artifact")
                data = self.get_artifact(record["data_hash"])
                results.append({**record, "data": data, "valid": current, "current": current,
                                "invalidated_reason": None if current else "commitment_superseded"})
            return results
