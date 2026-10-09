"""Protocol and replay tests use an isolated fake endpoint, never local weights."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import tempfile
import threading
import time
import unittest

from fev.model_adapter import (DEFAULT_MODEL, ModelAdapterError, OllamaProposer,
                               ReplayProposer, proposal_schema)


@contextmanager
def endpoint(chat_response=None, show_extra=None, delay=0):
    state = {"requests": [], "digest": "digest-A", "active": 0, "peak": 0,
             "chat_response": chat_response or {"message": {"content": '{"forks":[]}'},
                 "done": True, "done_reason": "stop", "eval_count": 9,
                 "eval_duration": 1000, "prompt_eval_count": 30}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.handle_api()

        def do_POST(self):
            self.handle_api()

        def handle_api(self):
            count = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(count)) if count else None
            state["requests"].append((self.path, body))
            status = 200
            if self.path == "/api/tags":
                result = {"models": [{"name": DEFAULT_MODEL, "digest": state["digest"]}]}
            elif self.path == "/api/version":
                result = {"version": "test-0.34.0"}
            elif self.path == "/api/show":
                result = {"template": "{{ .Prompt }}", "parameters": "num_ctx 16384",
                          "modelfile": "FROM /models/sha256-real\nRENDERER qwen3.8\nPARSER qwen3.5\n",
                          "model_info": {"tokenizer.ggml.model": "gpt2"}, **(show_extra or {})}
            elif self.path == "/api/chat":
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                time.sleep(delay)
                result = deepcopy(state["chat_response"])
                if isinstance(result, tuple):
                    status, result = result
                state["active"] -= 1
            else:
                status, result = 404, {"error": "missing"}
            payload = result if isinstance(result, str) else json.dumps(result)
            payload = payload.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            yield state, OllamaProposer(url=f"http://127.0.0.1:{server.server_port}",
                                       lock_dir=directory)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ModelAdapterTests(unittest.TestCase):
    def test_success_pins_real_metadata_and_structured_request(self):
        with endpoint() as (state, proposer):
            record = proposer.propose("proposal", {"raw_text": "Return a list."}, 100)
            self.assertEqual(record["data"], {"forks": []})
            self.assertEqual(record["generated_tokens"], 9)
            self.assertEqual(record["manifest"]["model_digest"], "digest-A")
            self.assertEqual(record["manifest"]["runtime"]["version"], "test-0.34.0")
            self.assertEqual(record["manifest"]["installed_model_blob"], "/models/sha256-real")
            self.assertEqual(record["manifest"]["renderer"], "qwen3.8")
            self.assertEqual(record["manifest"]["tokenizer"], {"tokenizer.ggml.model": "gpt2"})
            body = next(body for path, body in state["requests"] if path == "/api/chat")
            self.assertFalse(body["think"])
            self.assertFalse(body["stream"])
            self.assertEqual(body["options"]["num_predict"], 100)
            self.assertEqual(body["format"], proposal_schema("proposal"))

    def test_truncation_rejected_even_when_json_parses(self):
        with endpoint() as (state, proposer):
            state["chat_response"]["done_reason"] = "length"
            with self.assertRaises(ModelAdapterError) as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.record["raw_response"], '{"forks":[]}')
            self.assertEqual(caught.exception.generated_tokens, 9)
            self.assertIn("Incomplete", str(caught.exception))

    def test_missing_token_count_charges_reserved_budget(self):
        with endpoint() as (state, proposer):
            del state["chat_response"]["eval_count"]
            with self.assertRaises(ModelAdapterError) as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.generated_tokens, 100)
            self.assertFalse(caught.exception.record["token_count_known"])

    def test_empty_completed_output_retains_known_token_count(self):
        with endpoint() as (state, proposer):
            state["chat_response"]["message"]["content"] = ""
            state["chat_response"]["eval_count"] = 0
            with self.assertRaises(ModelAdapterError) as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.generated_tokens, 0)
            self.assertTrue(caught.exception.record["token_count_known"])

    def test_invalid_json_and_provenance_never_accepted(self):
        for raw in ('{"forks":[]', '{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}',
                    '{"a":"\\ud800"}', '{"a":' + '[' * 1100 + '0' + ']' * 1100 + '}',
                    '{"claims":[{"provenance":"user_confirmed"}]}'):
            with self.subTest(raw=raw), endpoint() as (state, proposer):
                state["chat_response"]["message"]["content"] = raw
                with self.assertRaises(ModelAdapterError) as caught:
                    proposer.propose("proposal", {}, 100)
                self.assertEqual(caught.exception.record["raw_response"], raw)
                self.assertEqual(caught.exception.generated_tokens, 9)

    def test_proposal_transport_depth_is_bounded_below_parser_recursion_limit(self):
        with endpoint() as (state, proposer):
            state["chat_response"]["message"]["content"] = '{"a":' + '[' * 70 + '0' + ']' * 70 + '}'
            with self.assertRaisesRegex(ModelAdapterError, "transport depth") as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.generated_tokens, 9)

    def test_context_guard_retains_complete_unicode_request_without_dispatch(self):
        with endpoint() as (state, proposer):
            source = "文" * 5000
            with self.assertRaisesRegex(ModelAdapterError, "context budget") as caught:
                proposer.propose("proposal", {"raw_text": source}, 100)
            self.assertEqual(caught.exception.generated_tokens, 0)
            record = caught.exception.record
            self.assertEqual(json.loads(record["request"]["messages"][-1]["content"])["packet"]["raw_text"], source)
            self.assertGreater(record["context_budget"]["input_utf8_bytes"], 15000)
            self.assertFalse(any(path == "/api/chat" for path, _ in state["requests"]))

    def test_context_and_timeout_configuration_rejects_invalid_numbers(self):
        for num_ctx in (False, "8192", 0, 262145):
            with self.subTest(num_ctx=num_ctx), self.assertRaises(ValueError):
                OllamaProposer(options={"num_ctx": num_ctx})
        for timeout in (False, "180", 0, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                OllamaProposer(timeout=timeout)

    def test_generation_options_are_pinned_to_manifest(self):
        with endpoint() as (state, proposer):
            proposer.manifest()
            proposer.options["seed"] = 99
            with self.assertRaisesRegex(ModelAdapterError, "options changed") as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.generated_tokens, 0)
            self.assertFalse(any(path == "/api/chat" for path, _ in state["requests"]))

    def test_runtime_drift_before_and_after_generation_preserves_token_cost(self):
        for when, expected_tokens in (("before", 0), ("after", 9)):
            with self.subTest(when=when), endpoint() as (_, proposer):
                proposer.manifest()
                actual_request = proposer._request
                version_calls = []

                def request(path, body=None):
                    data, raw = actual_request(path, body)
                    if path == "/api/version":
                        version_calls.append(path)
                        if when == "before" or len(version_calls) > 1:
                            data["version"] = "changed-runtime"
                    return data, raw

                proposer._request = request
                with self.assertRaisesRegex(ModelAdapterError, "runtime changed") as caught:
                    proposer.propose("proposal", {}, 100)
                self.assertEqual(caught.exception.generated_tokens, expected_tokens)

    def test_http_error_retains_raw_and_conservative_token_cost(self):
        with endpoint((500, {"error": "runtime failed"})) as (_, proposer):
            with self.assertRaises(ModelAdapterError) as caught:
                proposer.propose("proposal", {}, 100)
            self.assertIn("runtime failed", caught.exception.record["raw_api_response"])
            self.assertEqual(caught.exception.generated_tokens, 100)

    def test_timeout_charges_budget_and_preserves_manifest(self):
        with endpoint(delay=0.2) as (_, proposer):
            proposer.manifest()
            proposer.timeout = 0.03
            with self.assertRaises(ModelAdapterError) as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.generated_tokens, 100)
            self.assertIn("manifest", caught.exception.record)

    def test_mutated_tag_blocks_dispatch(self):
        with endpoint() as (state, proposer):
            proposer.manifest()
            state["digest"] = "digest-B"
            with self.assertRaisesRegex(ModelAdapterError, "tag changed") as caught:
                proposer.propose("proposal", {}, 100)
            self.assertEqual(caught.exception.generated_tokens, 0)
            self.assertFalse(any(path == "/api/chat" for path, _ in state["requests"]))

    def test_cloud_backed_model_rejected(self):
        with endpoint(show_extra={"remote_host": "https://ollama.com"}) as (_, proposer):
            with self.assertRaisesRegex(ModelAdapterError, "Cloud"):
                proposer.manifest()

    def test_concurrent_calls_serialize(self):
        with endpoint(delay=0.04) as (state, proposer):
            proposer.manifest()
            with ThreadPoolExecutor(max_workers=3) as executor:
                records = list(executor.map(lambda _: proposer.propose("proposal", {}, 100), range(3)))
            self.assertEqual(len(records), 3)
            self.assertEqual(state["peak"], 1)

    def test_remote_urls_and_invalid_stages_rejected(self):
        for url in ("https://127.0.0.1:11435", "http://example.com", "http://127.0.0.1/api",
                    "http://user:password@localhost:11435", "http://localhost:11435?x=y"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaProposer(url=url)
        with self.assertRaises(ValueError):
            proposal_schema("invented")

    def test_replay_is_offline_and_binding_sensitive(self):
        with endpoint() as (_, proposer):
            packet = {"raw_text": "Source"}
            original = proposer.propose("proposal", packet, 100)
        replay = ReplayProposer([original])
        with self.assertRaisesRegex(ModelAdapterError, "stage mismatch"):
            replay.propose("critic", packet, 100)
        with self.assertRaisesRegex(ModelAdapterError, "task binding mismatch"):
            replay.propose("proposal", {"raw_text": "Changed"}, 100)
        with self.assertRaisesRegex(ModelAdapterError, "token cap mismatch"):
            replay.propose("proposal", packet, 99)
        result = replay.propose("proposal", packet, 100)
        self.assertEqual(result["data"], original["data"])
        self.assertEqual(result["generated_tokens"], original["generated_tokens"])
        with self.assertRaisesRegex(ModelAdapterError, "exhausted"):
            replay.propose("proposal", packet, 100)

    def test_replay_allows_feedback_drift_but_binds_source_scope_and_language(self):
        packet = {"raw_text": "Source", "scope": {"max_length": 4}, "language": "v1",
                  "context": {"choice": 1}, "retained": [{"elapsed_ms": 17}]}
        with endpoint() as (_, proposer):
            original = proposer.propose("critic", packet, 100)
        drifted = {**packet, "retained": [{"elapsed_ms": 19}]}
        result = ReplayProposer([original]).propose("critic", drifted, 100)
        self.assertTrue(result["replay_feedback_drift"])
        self.assertEqual(result["replay_binding"], "task_and_stage")
        with self.assertRaisesRegex(ModelAdapterError, "packet hash mismatch"):
            ReplayProposer([original], strict_packet=True).propose("critic", drifted, 100)
        for key, replacement in (("scope", {"max_length": 5}), ("language", "v2"),
                                 ("context", {"choice": 2})):
            with self.subTest(key=key), self.assertRaisesRegex(ModelAdapterError, "task binding"):
                ReplayProposer([original]).propose("critic", {**packet, key: replacement}, 100)

    def test_replay_errors_and_raw_parsed_disagreement(self):
        replay = ReplayProposer([{"stage": "proposal", "error": "truncated", "generated_tokens": 20}])
        with self.assertRaises(ModelAdapterError) as caught:
            replay.propose("proposal", {}, 100)
        self.assertEqual(caught.exception.generated_tokens, 20)
        replay = ReplayProposer([{"data": {"forks": []}, "raw_response": '{"forks":[1]}'}])
        with self.assertRaisesRegex(ModelAdapterError, "differs"):
            replay.propose("proposal", {}, 100)

    def test_replay_error_records_cannot_bypass_token_count_validation(self):
        for count in (-1, "unknown", True):
            with self.subTest(count=count):
                replay = ReplayProposer([{"error": "failed", "generated_tokens": count}])
                with self.assertRaisesRegex(ModelAdapterError, "Invalid recorded token count") as caught:
                    replay.propose("proposal", {}, 100)
                self.assertEqual(caught.exception.generated_tokens, 100)
                self.assertFalse(caught.exception.record["token_count_known"])
        with self.assertRaisesRegex(ValueError, "JSON objects"):
            ReplayProposer(["not a record"])


if __name__ == "__main__":
    unittest.main()
