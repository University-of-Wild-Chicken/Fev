"""Persisted end-to-end workflow across independent CLI processes."""
import json
from pathlib import Path
import subprocess
import sys


def test_cli_replay_selection_execution_and_stale_rejection(tmp_path):
    root = Path(__file__).resolve().parents[1]

    def cli(*args, success=True):
        process = subprocess.run([sys.executable, "-m", "fev", "--ledger", str(tmp_path / "ledger"), *args],
                                 cwd=root, text=True, capture_output=True, timeout=30)
        assert (process.returncode == 0) == success, process.stderr
        return json.loads(process.stdout if success else process.stderr)

    result = cli("analyze", "Return the second-largest number.", "--replay", str(root / "examples/second_largest.replay.json"))
    assert result["decision"] == "clarify"
    selected = cli("select", result["request_id"], "F1", "--reason", "Synthetic CLI test user selects distinct", "--run-hash", result["run_hash"])
    output = cli("run", selected["commitment_id"], "[5,5,3]")
    assert output["result"] == {"kind": "value", "value": 3}
    replacement = cli("select", result["request_id"], "F2", "--reason", "Synthetic CLI test user selects multiplicity")
    assert cli("run", replacement["commitment_id"], "[5,5,3]")["result"]["value"] == 5
    assert cli("run", selected["commitment_id"], "[5,5,3]", success=False)["error"] == "StaleCommitmentError"
    assert "outside the committed scope" in cli("run", replacement["commitment_id"], "[99,98]", success=False)["message"]
