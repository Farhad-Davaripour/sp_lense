from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).with_name("submit_openai_judge_requests.py")
SPEC = importlib.util.spec_from_file_location("submit_openai_judge_requests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TRANSPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRANSPORT)


def _request() -> dict:
    request = {
        "model": TRANSPORT.LOCKED_MODEL,
        "input": [
            {"role": "system", "content": "Score the response."},
            {"role": "user", "content": "Return JSON."},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 128,
        "store": False,
    }
    key = {
        "instruction_pair_id": "pair_001",
        "question_id": "question_001",
        "rollout_index": 0,
        "polarity": "positive",
    }
    prompt_hash = TRANSPORT._sha256(request["input"])
    config_hash = TRANSPORT.LOCKED_PERSONA_CONFIG_SHA256
    request_id = TRANSPORT._sha256(
        {
            "schema_version": TRANSPORT.PERSONA_EXCHANGE_SCHEMA,
            "key": key,
            "judge_prompt_sha256": prompt_hash,
            "judge_config_sha256": config_hash,
        }
    )
    return {
        "schema_version": TRANSPORT.PERSONA_EXCHANGE_SCHEMA,
        "request_id": request_id,
        "key": key,
        "judge_prompt_sha256": prompt_hash,
        "judge_config_sha256": config_hash,
        "request": request,
    }


def _open_request() -> dict:
    request = {
        "model": TRANSPORT.LOCKED_MODEL,
        "input": [
            {"role": "system", "content": "Classify the response."},
            {"role": "user", "content": "Return the locked JSON object."},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 256,
        "store": False,
        "tools": [],
        "truncation": "disabled",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "open_behavior_judgment",
                "strict": True,
                "schema": TRANSPORT.OPEN_RESPONSE_SCHEMA,
            }
        },
    }
    return {
        "schema_version": TRANSPORT.OPEN_EXCHANGE_SCHEMA,
        "request_id": "b" * 64,
        "generation_sha256s": ["c" * 64],
        "judge_prompt_sha256": TRANSPORT._sha256(request["input"]),
        "judge_request_content_sha256": TRANSPORT._sha256(
            {
                "schema_version": TRANSPORT.OPEN_REQUEST_CONTENT_SCHEMA,
                "request": request,
            }
        ),
        "judge_config_sha256": TRANSPORT.LOCKED_OPEN_CONFIG_SHA256,
        "judge_protocol_sha256": TRANSPORT.LOCKED_OPEN_PROTOCOL_SHA256,
        "request": request,
    }


class JudgeTransportTests(unittest.TestCase):
    def test_process_environment_key_takes_precedence(self) -> None:
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            self.assertEqual(TRANSPORT._load_api_key(), "test-key")

    def test_output_text_extraction(self) -> None:
        response = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": '{"trait_score":50}'},
                    ]
                }
            ]
        }
        self.assertEqual(TRANSPORT._extract_output_text(response), '{"trait_score":50}')

    def test_cost_bound_is_positive_and_conservative(self) -> None:
        estimate = TRANSPORT._upper_bound_cost(
            [_request()],
            input_price_per_million=0.40,
            output_price_per_million=1.60,
        )
        self.assertEqual(estimate["request_count"], 1)
        self.assertGreater(estimate["input_token_upper_bound"], 0)
        self.assertGreater(estimate["safe_upper_bound_usd"], estimate["raw_upper_bound_usd"])

    def test_judge_prices_are_exactly_locked(self) -> None:
        TRANSPORT._require_locked_prices(0.40, 1.60)
        with self.assertRaisesRegex(ValueError, "input price must equal the locked"):
            TRANSPORT._require_locked_prices(0.39, 1.60)
        with self.assertRaisesRegex(ValueError, "output price must equal the locked"):
            TRANSPORT._require_locked_prices(0.40, 1.59)

    def test_cost_preflight_is_immutable_after_submission_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir = root / "work"
            responses_path = root / "responses.jsonl"
            estimate = {
                "model": TRANSPORT.LOCKED_MODEL,
                "input_price_per_million_usd": 0.40,
                "output_price_per_million_usd": 1.60,
                "user_cost_ceiling_usd": 1.0,
            }
            TRANSPORT._publish_cost_preflight(work_dir, responses_path, estimate)
            attempt_dir = work_dir / "submission_attempts"
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "request.json").write_text("{}\n", encoding="utf-8")

            # An identical resume is accepted without rewriting the original record.
            original_bytes = (work_dir / "cost_preflight.json").read_bytes()
            TRANSPORT._publish_cost_preflight(work_dir, responses_path, dict(estimate))
            self.assertEqual((work_dir / "cost_preflight.json").read_bytes(), original_bytes)

            # Semantically equivalent JSON is still a mutation of the locked receipt.
            (work_dir / "cost_preflight.json").write_text(
                json.dumps(estimate, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "byte-for-byte immutable"):
                TRANSPORT._publish_cost_preflight(work_dir, responses_path, dict(estimate))
            (work_dir / "cost_preflight.json").write_bytes(original_bytes)

            changed = {**estimate, "user_cost_ceiling_usd": 2.0}
            with self.assertRaisesRegex(SystemExit, "byte-for-byte immutable"):
                TRANSPORT._publish_cost_preflight(work_dir, responses_path, changed)

    def test_submission_evidence_without_original_preflight_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir = root / "work"
            responses_path = root / "responses.jsonl"
            responses_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "without its original cost preflight"):
                TRANSPORT._publish_cost_preflight(
                    work_dir,
                    responses_path,
                    {
                        "model": TRANSPORT.LOCKED_MODEL,
                        "input_price_per_million_usd": 0.40,
                        "output_price_per_million_usd": 1.60,
                        "user_cost_ceiling_usd": 1.0,
                    },
                )

    def test_wrong_model_is_rejected(self) -> None:
        row = _request()
        row["request"]["model"] = "gpt-4.1-mini"
        with self.assertRaisesRegex(ValueError, "not 'gpt-4.1-mini-2025-04-14'"):
            TRANSPORT._validate_requests([row])

    def test_request_metadata_and_payload_bindings_are_recomputed(self) -> None:
        persona = _request()
        opened = _open_request()
        TRANSPORT._validate_requests([persona])
        TRANSPORT._validate_requests([opened])

        tampered_persona = json.loads(json.dumps(persona))
        tampered_persona["request"]["input"][1]["content"] += " tampered"
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            TRANSPORT._validate_requests([tampered_persona])

        tampered_open = json.loads(json.dumps(opened))
        tampered_open["request"]["text"]["format"]["strict"] = False
        with self.assertRaisesRegex(ValueError, "structured-output schema"):
            TRANSPORT._validate_requests([tampered_open])

    def test_client_request_id_is_deterministic_ascii(self) -> None:
        row = _request()
        first = TRANSPORT._client_request_id(row["request_id"], row["request"])
        second = TRANSPORT._client_request_id(row["request_id"], row["request"])
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("sp-lense-"))
        self.assertLessEqual(len(first), 512)
        first.encode("ascii", errors="strict")

    def test_network_failure_is_not_automatically_retried(self) -> None:
        row = _request()
        with (
            mock.patch.object(
                TRANSPORT.urllib.request,
                "urlopen",
                side_effect=TRANSPORT.urllib.error.URLError("connection lost"),
            ) as urlopen,
            self.assertRaises(TRANSPORT.AmbiguousSubmissionError),
        ):
            TRANSPORT._post_response(
                row["request"],
                "test-key",
                client_request_id="sp-lense-test",
            )
        self.assertEqual(urlopen.call_count, 1)

    def test_server_error_is_not_automatically_retried(self) -> None:
        row = _request()
        error = TRANSPORT.urllib.error.HTTPError(
            TRANSPORT.API_URL,
            500,
            "server error",
            {},
            io.BytesIO(b'{"error":"internal"}'),
        )
        with (
            mock.patch.object(TRANSPORT.urllib.request, "urlopen", side_effect=error) as urlopen,
            self.assertRaises(TRANSPORT.AmbiguousSubmissionError),
        ):
            TRANSPORT._post_response(
                row["request"],
                "test-key",
                client_request_id="sp-lense-test",
            )
        self.assertEqual(urlopen.call_count, 1)

    def test_rate_limit_retries_only_explicit_rejections(self) -> None:
        row = _request()

        def rate_limit_error() -> Exception:
            return TRANSPORT.urllib.error.HTTPError(
                TRANSPORT.API_URL,
                429,
                "rate limited",
                {"Retry-After": "0"},
                io.BytesIO(b'{"error":"rate_limited"}'),
            )

        with (
            mock.patch.object(TRANSPORT, "MAX_RATE_LIMIT_RETRIES", 1),
            mock.patch.object(TRANSPORT.time, "sleep"),
            mock.patch.object(
                TRANSPORT.urllib.request,
                "urlopen",
                side_effect=[rate_limit_error(), rate_limit_error()],
            ) as urlopen,
            self.assertRaises(TRANSPORT.RejectedSubmissionError),
        ):
            TRANSPORT._post_response(
                row["request"],
                "test-key",
                client_request_id="sp-lense-test",
            )
        self.assertEqual(urlopen.call_count, 2)

    def test_attempt_record_binds_request_and_state(self) -> None:
        row = _request()
        client_request_id = TRANSPORT._client_request_id(row["request_id"], row["request"])
        with tempfile.TemporaryDirectory() as directory:
            attempt_path = Path(directory) / "attempt.json"
            TRANSPORT._atomic_json(
                attempt_path,
                TRANSPORT._attempt_payload(
                    row,
                    client_request_id=client_request_id,
                    state="ambiguous_blocked",
                ),
            )
            self.assertEqual(
                TRANSPORT._validate_prior_attempt(
                    attempt_path,
                    row,
                    client_request_id=client_request_id,
                ),
                "ambiguous_blocked",
            )

    def test_transport_lock_rejects_concurrent_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / "work"
            first = TRANSPORT._acquire_transport_lock(work_dir)
            try:
                with self.assertRaisesRegex(SystemExit, "exclusive lock"):
                    TRANSPORT._acquire_transport_lock(work_dir)
            finally:
                first.close()
            third = TRANSPORT._acquire_transport_lock(work_dir)
            third.close()

    def test_ambiguous_attempt_blocks_second_main_submission(self) -> None:
        row = _request()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            work_dir = root / "work"
            requests_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            arguments = [
                "submit_openai_judge_requests.py",
                "--requests",
                str(requests_path),
                "--responses",
                str(responses_path),
                "--work-dir",
                str(work_dir),
                "--max-cost-usd",
                "1",
            ]
            with (
                mock.patch.object(TRANSPORT.sys, "argv", arguments),
                mock.patch.object(TRANSPORT, "_load_api_key", return_value="test-key"),
                mock.patch.object(
                    TRANSPORT,
                    "_post_response",
                    side_effect=TRANSPORT.AmbiguousSubmissionError("unknown outcome"),
                ) as post_response,
            ):
                with self.assertRaises(TRANSPORT.AmbiguousSubmissionError):
                    TRANSPORT.main()
                with self.assertRaisesRegex(SystemExit, "Automatic resubmission is blocked"):
                    TRANSPORT.main()
            self.assertEqual(post_response.call_count, 1)
            attempt_path = work_dir / "submission_attempts" / f"{row['request_id']}.json"
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            self.assertEqual(attempt["state"], "ambiguous_blocked")
            self.assertFalse(responses_path.exists())

    def test_invalid_success_is_quarantined_and_blocks_resubmission(self) -> None:
        row = _request()
        response = {
            "id": "resp_missing_trace",
            "model": TRANSPORT.LOCKED_MODEL,
            "status": "completed",
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"trait_score":50,"coherence_score":90}',
                        }
                    ]
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }
        headers = {
            "x_client_request_id": TRANSPORT._client_request_id(row["request_id"], row["request"]),
            "x_request_id": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            work_dir = root / "work"
            requests_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            arguments = [
                "submit_openai_judge_requests.py",
                "--requests",
                str(requests_path),
                "--responses",
                str(responses_path),
                "--work-dir",
                str(work_dir),
                "--max-cost-usd",
                "1",
            ]
            with (
                mock.patch.object(TRANSPORT.sys, "argv", arguments),
                mock.patch.object(TRANSPORT, "_load_api_key", return_value="test-key"),
                mock.patch.object(
                    TRANSPORT,
                    "_post_response",
                    return_value=(response, headers),
                ) as post_response,
            ):
                with self.assertRaisesRegex(ValueError, "x-request-id"):
                    TRANSPORT.main()
                with self.assertRaisesRegex(SystemExit, "Automatic resubmission is blocked"):
                    TRANSPORT.main()
            self.assertEqual(post_response.call_count, 1)
            attempt = json.loads(
                (work_dir / "submission_attempts" / f"{row['request_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(attempt["state"], "response_validation_blocked")
            quarantine = json.loads(
                (work_dir / "blocked_responses" / f"{row['request_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(quarantine["api_response"], response)
            self.assertFalse(responses_path.exists())

    def test_successful_main_run_is_accepted_by_verify_only(self) -> None:
        row = _request()
        raw_response = '{"trait_score":50,"coherence_score":90}'
        response = {
            "id": "resp_success_001",
            "model": TRANSPORT.LOCKED_MODEL,
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": raw_response}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }
        response_headers = {
            "x_client_request_id": TRANSPORT._client_request_id(row["request_id"], row["request"]),
            "x_request_id": "req_success_001",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            work_dir = root / "work"
            requests_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            submit_arguments = [
                "submit_openai_judge_requests.py",
                "--requests",
                str(requests_path),
                "--responses",
                str(responses_path),
                "--work-dir",
                str(work_dir),
                "--max-cost-usd",
                "1",
            ]
            with (
                mock.patch.object(TRANSPORT.sys, "argv", submit_arguments),
                mock.patch.object(TRANSPORT, "_load_api_key", return_value="test-key"),
                mock.patch.object(
                    TRANSPORT,
                    "_post_response",
                    return_value=(response, response_headers),
                ) as post_response,
            ):
                self.assertEqual(TRANSPORT.main(), 0)
            self.assertEqual(post_response.call_count, 1)

            verify_arguments = [
                "submit_openai_judge_requests.py",
                "--requests",
                str(requests_path),
                "--responses",
                str(responses_path),
                "--work-dir",
                str(work_dir),
                "--verify-only",
            ]
            with (
                mock.patch.object(TRANSPORT.sys, "argv", verify_arguments),
                mock.patch.object(
                    TRANSPORT,
                    "_post_response",
                    side_effect=AssertionError("verify-only attempted a network call"),
                ) as verify_post,
            ):
                self.assertEqual(TRANSPORT.main(), 0)
            self.assertEqual(verify_post.call_count, 0)
            attempt = json.loads(
                (work_dir / "submission_attempts" / f"{row['request_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(attempt["state"], "completed")

    def test_resume_recovers_prepared_attempt_from_fully_validated_receipt(self) -> None:
        row = _request()
        raw_response = '{"trait_score":50,"coherence_score":90}'
        client_request_id = TRANSPORT._client_request_id(row["request_id"], row["request"])
        api_response = {
            "id": "resp_crash_recovery_001",
            "model": TRANSPORT.LOCKED_MODEL,
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": raw_response}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }
        receipt = {
            "schema_version": "sp_lense.openai_judge_api_receipt.v1",
            "request_id": row["request_id"],
            "request_sha256": TRANSPORT._sha256(row["request"]),
            "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            "api_response_id": api_response["id"],
            "api_model": api_response["model"],
            "api_status": api_response["status"],
            "x_client_request_id": client_request_id,
            "x_request_id": "req_crash_recovery_001",
            "usage": api_response["usage"],
            "api_response_sha256": TRANSPORT._sha256(api_response),
            "raw_response": raw_response,
            "api_response": api_response,
        }
        for shard_already_written in (False, True):
            with (
                self.subTest(shard_already_written=shard_already_written),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                requests_path = root / "requests.jsonl"
                responses_path = root / "responses.jsonl"
                work_dir = root / "work"
                requests_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                arguments = [
                    "submit_openai_judge_requests.py",
                    "--requests",
                    str(requests_path),
                    "--responses",
                    str(responses_path),
                    "--work-dir",
                    str(work_dir),
                    "--max-cost-usd",
                    "1",
                ]
                with mock.patch.object(TRANSPORT.sys, "argv", [*arguments, "--dry-run"]):
                    self.assertEqual(TRANSPORT.main(), 0)

                receipt_path = work_dir / "api_receipts" / f"{row['request_id']}.json"
                attempt_path = work_dir / "submission_attempts" / f"{row['request_id']}.json"
                shard_path = work_dir / "response_shards" / f"{row['request_id']}.json"
                TRANSPORT._atomic_json(receipt_path, receipt)
                TRANSPORT._atomic_json(
                    attempt_path,
                    TRANSPORT._attempt_payload(
                        row,
                        client_request_id=client_request_id,
                        state="prepared",
                    ),
                )
                if shard_already_written:
                    TRANSPORT._atomic_json(
                        shard_path,
                        {
                            "schema_version": row["schema_version"],
                            "request_id": row["request_id"],
                            "raw_response": raw_response,
                        },
                    )

                with (
                    mock.patch.object(TRANSPORT.sys, "argv", arguments),
                    mock.patch.object(
                        TRANSPORT,
                        "_load_api_key",
                        side_effect=AssertionError("receipt recovery requested an API key"),
                    ) as load_api_key,
                    mock.patch.object(
                        TRANSPORT,
                        "_post_response",
                        side_effect=AssertionError("recovery attempted a network call"),
                    ) as post_response,
                ):
                    self.assertEqual(TRANSPORT.main(), 0)
                self.assertEqual(load_api_key.call_count, 0)
                self.assertEqual(post_response.call_count, 0)
                recovered_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                self.assertEqual(recovered_attempt["state"], "completed")
                self.assertEqual(recovered_attempt["x_request_id"], receipt["x_request_id"])
                self.assertTrue(shard_path.is_file())
                self.assertTrue(responses_path.is_file())

    def test_missing_api_key_does_not_create_prepared_attempt(self) -> None:
        row = _request()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            responses_path = root / "responses.jsonl"
            work_dir = root / "work"
            requests_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            arguments = [
                "submit_openai_judge_requests.py",
                "--requests",
                str(requests_path),
                "--responses",
                str(responses_path),
                "--work-dir",
                str(work_dir),
                "--max-cost-usd",
                "1",
            ]
            with (
                mock.patch.object(TRANSPORT.sys, "argv", arguments),
                mock.patch.object(TRANSPORT, "_load_api_key", return_value=None),
                mock.patch.object(
                    TRANSPORT,
                    "_post_response",
                    side_effect=AssertionError("missing-key run attempted a network call"),
                ) as post_response,
                self.assertRaisesRegex(SystemExit, "OPENAI_API_KEY is not set"),
            ):
                TRANSPORT.main()
            self.assertEqual(post_response.call_count, 0)
            attempt_path = work_dir / "submission_attempts" / f"{row['request_id']}.json"
            self.assertFalse(attempt_path.exists())

    def test_resume_rejects_invalid_receipt_without_promoting_attempt(self) -> None:
        row = _request()
        raw_response = '{"trait_score":50,"coherence_score":90}'
        client_request_id = TRANSPORT._client_request_id(row["request_id"], row["request"])
        api_response = {
            "id": "resp_invalid_recovery_001",
            "model": "wrong-model",
            "status": "completed",
            "output": [{"content": [{"type": "output_text", "text": raw_response}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            attempt_path = root / "attempt.json"
            receipt = {
                "schema_version": "sp_lense.openai_judge_api_receipt.v1",
                "request_id": row["request_id"],
                "request_sha256": TRANSPORT._sha256(row["request"]),
                "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                "api_response_id": api_response["id"],
                "api_model": api_response["model"],
                "api_status": api_response["status"],
                "x_client_request_id": client_request_id,
                "x_request_id": "req_invalid_recovery_001",
                "usage": api_response["usage"],
                "api_response_sha256": TRANSPORT._sha256(api_response),
                "raw_response": raw_response,
                "api_response": api_response,
            }
            TRANSPORT._atomic_json(receipt_path, receipt)
            TRANSPORT._atomic_json(
                attempt_path,
                TRANSPORT._attempt_payload(
                    row,
                    client_request_id=client_request_id,
                    state="prepared",
                ),
            )
            with self.assertRaisesRegex(ValueError, "wrong-model"):
                validated = TRANSPORT._validate_completed_receipt(
                    receipt_path,
                    row,
                    client_request_id=client_request_id,
                )
                TRANSPORT._recover_completed_attempt(
                    attempt_path,
                    row,
                    validated,
                    client_request_id=client_request_id,
                )
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            self.assertEqual(attempt["state"], "prepared")

    def test_receipt_reconstructs_interrupted_response_shard(self) -> None:
        row = _request()
        raw_response = '{"trait_score":50,"coherence_score":90}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "receipt.json"
            shard_path = root / "shard.json"
            receipt = {
                "request_sha256": TRANSPORT._sha256(row["request"]),
                "raw_response": raw_response,
                "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            }
            TRANSPORT._atomic_json(receipt_path, receipt)
            self.assertTrue(TRANSPORT._existing_response(shard_path, receipt_path, row))
            reconstructed = json.loads(shard_path.read_text(encoding="utf-8"))
            self.assertEqual(reconstructed["raw_response"], raw_response)

    def test_verify_completed_transport_checks_receipt_and_attempt_sets(self) -> None:
        row = _request()
        raw_response = '{"trait_score":50,"coherence_score":90}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir = root / "work"
            shard_path = work_dir / "response_shards" / f"{row['request_id']}.json"
            receipt_path = work_dir / "api_receipts" / f"{row['request_id']}.json"
            attempt_path = work_dir / "submission_attempts" / f"{row['request_id']}.json"
            responses_path = root / "responses.jsonl"
            client_request_id = TRANSPORT._client_request_id(row["request_id"], row["request"])
            api_response = {
                "id": "resp_test_001",
                "model": TRANSPORT.LOCKED_MODEL,
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": raw_response}]}],
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            }
            shard = {
                "schema_version": row["schema_version"],
                "request_id": row["request_id"],
                "raw_response": raw_response,
            }
            receipt = {
                "schema_version": "sp_lense.openai_judge_api_receipt.v1",
                "request_id": row["request_id"],
                "request_sha256": TRANSPORT._sha256(row["request"]),
                "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                "api_response_id": api_response["id"],
                "api_model": api_response["model"],
                "api_status": api_response["status"],
                "x_client_request_id": client_request_id,
                "x_request_id": "req_test_001",
                "usage": api_response["usage"],
                "api_response_sha256": TRANSPORT._sha256(api_response),
                "raw_response": raw_response,
                "api_response": api_response,
            }
            attempt = TRANSPORT._attempt_payload(
                row,
                client_request_id=client_request_id,
                state="completed",
                server_request_id="req_test_001",
            )
            TRANSPORT._atomic_json(shard_path, shard)
            TRANSPORT._atomic_json(receipt_path, receipt)
            TRANSPORT._atomic_json(attempt_path, attempt)
            requests_file_sha256 = "a" * 64
            preflight = TRANSPORT._upper_bound_cost(
                [row],
                input_price_per_million=0.40,
                output_price_per_million=1.60,
            )
            preflight.update(
                {
                    "model": TRANSPORT.LOCKED_MODEL,
                    "api_url": TRANSPORT.API_URL,
                    "input_price_per_million_usd": 0.40,
                    "output_price_per_million_usd": 1.60,
                    "user_cost_ceiling_usd": 1.0,
                    "requests_file_sha256": requests_file_sha256,
                }
            )
            TRANSPORT._atomic_json(work_dir / "cost_preflight.json", preflight)
            responses_path.write_text(
                json.dumps(shard, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            verified = TRANSPORT._verify_completed_transport(
                [row],
                requests_file_sha256=requests_file_sha256,
                responses_path=responses_path,
                work_dir=work_dir,
            )
            self.assertEqual(verified["request_count"], 1)
            self.assertEqual(verified["input_tokens"], 20)
            self.assertEqual(verified["output_tokens"], 10)

            attempt["state"] = "ambiguous_blocked"
            TRANSPORT._atomic_json(attempt_path, attempt)
            with self.assertRaisesRegex(ValueError, "attempt is not completed"):
                TRANSPORT._verify_completed_transport(
                    [row],
                    requests_file_sha256=requests_file_sha256,
                    responses_path=responses_path,
                    work_dir=work_dir,
                )


if __name__ == "__main__":
    unittest.main()
