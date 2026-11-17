"""Synthetic CPU fixtures only: no actual Game, model, network or checkpoint use."""
import copy
import json
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src import native_generation_transcript as transcript


CASE = "synthetic/neutral/7"
OTHER = "synthetic/base/7"
MESSAGES = [{"role": "system", "content": "synthetic system"},
            {"role": "user", "content": "synthetic input"}]
TOKENS = {1: "<think>", 2: "</think>", 3: "First thought.\n", 4: '{"action":"wait"}',
          5: "Second thought.\n", 6: '{"action":"stop"}', 8: "<|endoftext|>", 9: "<|im_end|>"}


class SyntheticTokenizer:
    def id_to_token(self, token):
        return TOKENS.get(token)

    def decode(self, ids, *, skip_special_tokens):
        if skip_special_tokens is not False:
            raise AssertionError("Special tags must remain visible")
        return "".join(TOKENS[token] for token in ids)


def reference():
    return transcript._Reference(SyntheticTokenizer(),
        lambda _: "synthetic verified prompt<|im_start|>assistant\n<think>\n",
        {"manifest_sha256": transcript.REFERENCE_MANIFEST_SHA256, "synthetic_fixture": True},
        [8, 9], 9, 10, 1, 2, "b" * 64)


def fixture_request(decision, *, case=CASE, generated=None, request_id=None, attempt=1):
    request_id = request_id or str(decision + 1).zfill(32)
    identity = {"case_id": case, "decision": decision, "attempt": attempt}
    messages = copy.deepcopy(MESSAGES)
    messages[-1]["content"] += " " + str(decision)
    digest = transcript._digest(messages)
    generated = generated if generated is not None else [3 if decision == 0 else 5, 2, 4 if decision == 0 else 6, 9]
    native = ("a" if decision == 0 else "c") * 32
    physical = {"request_id": request_id, "messages_sha256": digest,
        "base_manifest_sha256": "b" * 64, "vocab_size": 10,
        "tokenizer_metadata_sha256": transcript.NATIVE_METADATA_SHA256,
        "eos_token_ids": [8, 9], "canonical_eos_token_id": 9,
        "completion_tokens": len(generated), "actual_generated_token_ids": generated,
        "EOS_detected": generated[-1] in [8, 9], "finish_reason": "stop",
        "native_ticket_id": native, "completion_ack": {"request_id": native,
            "worker_started": True, "worker_finished": True, "backend_barrier_done": True,
            "post_checks_passed": True, "native_errors": []},
        "dispatch_record": {"phase": "GAME_WINDOW", "request_id": request_id, "window_sha256": "d" * 64}}
    guards = [{"event": "HTTP_ATTEMPT", "request_id": request_id, "messages_sha256": digest,
               "generation_identity": identity, "attempt": decision + 80,
               "window_sha256": "d" * 64, "exact_serialized_messages": messages},
              {"event": "POST_REQUEST_VALIDATED", "request_id": request_id, "messages_sha256": digest,
               "generation_identity": copy.deepcopy(identity), "window_sha256": "d" * 64,
               "decision": {"request_id": request_id, "post_request_completion_allowed": True},
               "fresh_health": {"last_request_evidence": physical}}]
    events = [{"type": "model_prompt", "case_id": case, "decision": decision, "user_prompt": "fixture"},
              {"type": "model_stream", "event": "generation_started", **identity},
              {"type": "model_stream", "event": "text_delta", **identity, "text": "client-visible fixture"},
              {"type": "model_stream", "event": "generation_completed", **identity, "stop_reason": "stop"}]
    return guards, events


class NativeTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.batch = self.base / "batch_01"
        self.batch.mkdir()
        self.guards, self.events = fixture_request(0)
        self.cases = [CASE]
        self.reference = reference()
        self.loader = patch.object(transcript, "_load_reference", side_effect=lambda: self.reference)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def save(self):
        for name, value in (("MANIFEST.json", {"samples": [{"case_id": case} for case in self.cases]}),
                            ("CLOSEOUT.json", {"results": [{"case_id": case, "status": "complete"} for case in self.cases]})):
            (self.batch / name).write_text(json.dumps(value), encoding="utf-8")
        for name, values in (("REQUEST_GUARD_EVIDENCE.jsonl", self.guards), ("EVENTS.jsonl", self.events)):
            (self.batch / name).write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")

    def document(self, case=CASE):
        self.save()
        return transcript.build_batch_transcripts(self.batch)[case]

    def generation(self):
        return self.document()["generations"][0]

    def physical(self):
        return self.guards[1]["fresh_health"]["last_request_evidence"]

    def publication(self):
        path = self.base / "canonical" / "hour" / "conversation" / "synthetic.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("- input: Existing canonical conversation\n", encoding="utf-8")
        raw_dir = path.parent.parent / "raw" / "synthetic"
        raw_dir.mkdir(parents=True)
        (raw_dir / "preserved.txt").write_text("raw preserved", encoding="utf-8")
        index = path.parents[2] / "INDEX.yaml"
        index.write_text("synthetic index preserved", encoding="utf-8")
        return {"case_id": CASE, "conversation": str(path), "raw_dir": str(raw_dir),
                "conversation_sha256": transcript._sha(path.read_bytes()),
                "source_event_sha256": transcript._sha((self.batch / "EVENTS.jsonl").read_bytes())}

    def test_complete_native_response_preserves_tags_and_separates_client_stream(self):
        document = self.document()
        self.assertEqual(document["status"], "complete")
        generation = document["generations"][0]
        self.assertEqual(generation["reasoning_text"], "First thought.\n")
        self.assertEqual(generation["action_text"], '{"action":"wait"}')
        self.assertEqual(generation["native_text_with_special_tokens"], 'First thought.\n</think>{"action":"wait"}<|im_end|>')
        self.assertEqual(generation["client_visible_stream"]["text"], "client-visible fixture")
        self.assertIn("internal intent", document["interpretation"])
        self.assertEqual(document["tokenizer"]["manifest_sha256"], transcript.REFERENCE_MANIFEST_SHA256)
        self.assertEqual(len(document["source_files"]["EVENTS.jsonl"]["sha256"]), 64)
        self.assertEqual(len(generation["guard_records"][0]["raw_record_sha256"]), 64)

    def test_reordered_multi_turn_and_multi_case_pair_only_by_explicit_identity(self):
        second_guards, second_events = fixture_request(1, attempt=2)
        other_guards, other_events = fixture_request(0, case=OTHER, request_id="e" * 32)
        self.guards = [second_guards[1], other_guards[0], self.guards[1], second_guards[0], self.guards[0], other_guards[1]]
        self.events = second_events + self.events + other_events
        self.cases.append(OTHER)
        documents = self.document()
        turns = documents["generations"]
        self.assertEqual([(row["decision"], row["attempt"], row["reasoning_text"]) for row in turns],
                         [(0, 1, "First thought.\n"), (1, 2, "Second thought.\n")])
        other = transcript.build_batch_transcripts(self.batch)[OTHER]
        self.assertEqual(other["generations"][0]["request_id"], "e" * 32)

    def test_guard_global_http_attempt_is_never_a_generation_identity(self):
        for row in self.guards:
            row.pop("generation_identity")
        generation = self.generation()
        self.assertEqual(generation["status"], "unavailable")
        self.assertNotIn("native_text_with_special_tokens", generation)
        self.assertIn("no order-based pairing", generation["issues"][0])

    def test_explicit_event_binding_can_supply_identity(self):
        for row in self.guards:
            row.pop("generation_identity")
        self.events.append({"type": "model_stream", "event": "native_request_bound",
                            "case_id": CASE, "decision": 0, "attempt": 1,
                            "request_id": self.guards[0]["request_id"],
                            "messages_sha256": self.guards[0]["messages_sha256"]})
        self.assertEqual(self.generation()["status"], "complete")

    def test_conflicting_identity_or_duplicate_request_is_unavailable(self):
        self.guards[1]["generation_identity"]["decision"] = 1
        self.assertEqual(self.generation()["status"], "unavailable")
        self.guards, self.events = fixture_request(0)
        second, _ = fixture_request(0, request_id="f" * 32)
        self.guards.extend(second)
        self.assertEqual(self.generation()["status"], "unavailable")

    def test_mismatched_request_messages_or_native_tokenizer_are_unavailable(self):
        mutations = [lambda: self.physical().update(request_id="f" * 32),
                     lambda: self.physical().update(messages_sha256="f" * 64),
                     lambda: self.physical().update(tokenizer_metadata_sha256="f" * 64),
                     lambda: self.guards[0]["exact_serialized_messages"][1].update(content="changed"),
                     lambda: self.guards.append(copy.deepcopy(self.guards[1]))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.guards, self.events = fixture_request(0)
                mutation()
                result = self.generation()
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("native_text_with_special_tokens", result)

    def test_absent_invalid_or_mismatched_ids_never_decode(self):
        for ids, count in ((None, 4), ([], 0), ([True, 9], 2), ([10, 9], 2), ([7, 9], 2), ([3, 2, 4, 9], 3)):
            with self.subTest(ids=ids, count=count):
                self.guards, self.events = fixture_request(0)
                self.physical().update(actual_generated_token_ids=ids, completion_tokens=count)
                result = self.generation()
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("reasoning_text", result)

    def test_eos_partial_and_incomplete_ack_remain_explicitly_incomplete(self):
        for changes in ({"actual_generated_token_ids": [3, 2, 4], "completion_tokens": 3, "EOS_detected": False, "finish_reason": "length"},
                        {"actual_generated_token_ids": [3, 9, 4, 9]},
                        {"EOS_detected": False}, {"finish_reason": "length"},
                        {"completion_ack": {}}):
            with self.subTest(changes=changes):
                self.guards, self.events = fixture_request(0)
                self.physical().update(changes)
                result = self.generation()
                self.assertEqual(result["status"], "incomplete")
                self.assertIn("native_text_with_special_tokens", result)
                self.assertNotIn("reasoning_text", result)

    def test_only_unambiguous_actual_tags_or_verified_prefill_split_reasoning(self):
        self.guards, self.events = fixture_request(0, generated=[1, 3, 2, 4, 9])
        self.guards[0].pop("exact_serialized_messages")
        self.assertEqual(self.generation()["reasoning_text"], "First thought.\n")
        for ids in ([3, 2, 4, 9], [3, 4, 9], [1, 3, 2, 2, 4, 9], [3, 1, 2, 4, 9]):
            with self.subTest(ids=ids):
                self.guards, self.events = fixture_request(0, generated=ids)
                self.guards[0].pop("exact_serialized_messages")
                result = self.generation()
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["reasoning_partition_status"], "unavailable")
                self.assertNotIn("reasoning_text", result)
        self.guards, self.events = fixture_request(0)
        self.reference.render_prompt = lambda _: "prompt without verified thinking prefix"
        self.assertNotIn("reasoning_text", self.generation())

    def test_missing_guard_or_tokenizer_marks_unavailable(self):
        self.save()
        (self.batch / "REQUEST_GUARD_EVIDENCE.jsonl").unlink()
        self.assertEqual(transcript.build_batch_transcripts(self.batch)[CASE]["status"], "incomplete_or_unavailable")
        self.save()
        with patch.object(transcript, "_load_reference", side_effect=OSError("synthetic missing tokenizer")):
            result = transcript.build_batch_transcripts(self.batch)[CASE]
        self.assertEqual(result["tokenizer"]["status"], "unavailable")
        self.assertEqual(result["generations"][0]["status"], "unavailable")

    def test_active_batch_and_source_drift_are_rejected(self):
        self.save()
        (self.batch / "CLOSEOUT.json").unlink()
        with self.assertRaisesRegex(transcript.TranscriptError, "active batches"):
            transcript.build_batch_transcripts(self.batch)
        self.save()
        def drift():
            (self.batch / "EVENTS.jsonl").write_bytes(b"changed source")
            return self.reference
        with patch.object(transcript, "_load_reference", side_effect=drift):
            with self.assertRaisesRegex(transcript.TranscriptError, "source changed"):
                transcript.build_batch_transcripts(self.batch)

    def test_writer_adds_companion_only_and_redelivery_does_not_overwrite(self):
        self.save()
        entry = self.publication()
        before = {path: path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        first = transcript.write_batch_transcripts(self.batch, [entry])
        artifact = Path(first["artifacts"][0]["path"])
        self.assertEqual(artifact.name, "synthetic.native-generation.yaml")
        self.assertEqual(yaml.safe_load(artifact.read_bytes())["status"], "complete")
        self.assertTrue(first["artifacts"][0]["created"])
        second = transcript.write_batch_transcripts(self.batch, [entry])
        self.assertFalse(second["artifacts"][0]["created"])
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(set(path for path in self.base.rglob("*") if path.is_file()) - set(before), {artifact})
        artifact.write_text("preserve conflicting companion", encoding="utf-8")
        with self.assertRaisesRegex(transcript.TranscriptError, "no overwrite"):
            transcript.write_batch_transcripts(self.batch, [entry])
        self.assertEqual(artifact.read_text(encoding="utf-8"), "preserve conflicting companion")

    def test_writer_requires_existing_conversation_and_exact_source_hash(self):
        self.save()
        entry = self.publication()
        entry["source_event_sha256"] = "f" * 64
        with self.assertRaisesRegex(transcript.TranscriptError, "canonical conversation"):
            transcript.write_batch_transcripts(self.batch, [entry])
        self.assertFalse(list(self.base.rglob("*.native-generation.yaml")))

    def test_subprocess_requires_explicit_cpu_venv_and_verifies_readback(self):
        self.save()
        entry = self.publication()
        with self.assertRaisesRegex(transcript.TranscriptError, "exact existing absolute"):
            transcript.run_decoder_subprocess(self.batch, [entry], "python")
        executable = Path(__import__("sys").executable).resolve()
        written = transcript.write_batch_transcripts(self.batch, [entry])
        with patch.object(transcript, "CPU_REFERENCE_PYTHON", executable), patch.object(
                transcript.subprocess, "run", return_value=CompletedProcess([], 0, json.dumps(written), "")) as run:
            result = transcript.run_decoder_subprocess(self.batch, [entry], executable)
            self.assertEqual(result["artifacts"][0]["status"], "complete")
            self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(json.loads(run.call_args.kwargs["input"]), [entry])
        with patch.object(transcript, "CPU_REFERENCE_PYTHON", executable), patch.object(
                transcript.subprocess, "run", return_value=CompletedProcess([], 1, "", "synthetic I/O failure")):
            with self.assertRaisesRegex(transcript.TranscriptError, "synthetic I/O failure"):
                transcript.run_decoder_subprocess(self.batch, [entry], executable)
        bad = copy.deepcopy(written)
        bad["artifacts"][0]["sha256"] = "f" * 64
        with patch.object(transcript, "CPU_REFERENCE_PYTHON", executable), patch.object(
                transcript.subprocess, "run", return_value=CompletedProcess([], 0, json.dumps(bad), "")):
            with self.assertRaisesRegex(transcript.TranscriptError, "readback differs"):
                transcript.run_decoder_subprocess(self.batch, [entry], executable)

    def test_cli_accepts_list_json_and_unavailable_evidence_is_success(self):
        from io import StringIO
        self.save()
        entry = self.publication()
        with patch.object(transcript, "_load_reference", side_effect=ImportError("synthetic CPU tokenizer missing")), patch(
                "sys.stdin", StringIO(json.dumps([entry]))), patch("sys.stdout", new_callable=StringIO) as stdout:
            transcript.main(["--batch", str(self.batch), "--entries-stdin"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["artifacts"][0]["status"], "incomplete_or_unavailable")
        document = yaml.safe_load(Path(result["artifacts"][0]["path"]).read_bytes())
        self.assertEqual(document["generations"][0]["status"], "unavailable")
        self.assertIn("synthetic CPU tokenizer missing", document["tokenizer"]["reason"])
        with self.assertRaisesRegex(transcript.TranscriptError, "Duplicate JSON key"):
            transcript._json('[{"case_id":"a","case_id":"b"}]')
        with self.assertRaisesRegex(transcript.TranscriptError, "Nonfinite JSON"):
            transcript._json('[{"value":NaN}]')


@unittest.skipUnless(transcript.CPU_REFERENCE_PYTHON.is_file(), "Existing offline CPU reference venv unavailable")
class NativeReferenceSubprocessTests(unittest.TestCase):
    def test_real_offline_cli_decodes_only_fabricated_closed_batch(self):
        """Real pinned tokenizer/CLI, wholly synthetic text and evidence, no model."""
        reference = transcript._load_reference()
        text = 'Synthetic recorded reasoning.\n</think>\n{"action":"wait"}'
        ids = reference.tokenizer.encode(text, add_special_tokens=False).ids + [reference.canonical_eos]
        guards, events = fixture_request(0, generated=ids)
        physical = guards[1]["fresh_health"]["last_request_evidence"]
        physical.update(base_manifest_sha256=reference.base_manifest_sha256,
                        vocab_size=reference.vocab_size, eos_token_ids=reference.eos_ids,
                        canonical_eos_token_id=reference.canonical_eos, EOS_detected=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "synthetic_batch"
            batch.mkdir()
            for name, value in (("MANIFEST.json", {"samples": [{"case_id": CASE}]}),
                                ("CLOSEOUT.json", {"results": [{"case_id": CASE, "status": "complete"}]})):
                (batch / name).write_text(json.dumps(value), encoding="utf-8")
            for name, rows in (("EVENTS.jsonl", events), ("REQUEST_GUARD_EVIDENCE.jsonl", guards)):
                (batch / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            conversation = root / "canonical" / "conversation" / "synthetic.yaml"
            conversation.parent.mkdir(parents=True)
            conversation.write_text("- input: synthetic\n", encoding="utf-8")
            entry = {"case_id": CASE, "conversation": str(conversation),
                     "conversation_sha256": transcript._sha(conversation.read_bytes()),
                     "raw_dir": str(root / "canonical" / "raw"),
                     "source_event_sha256": transcript._sha((batch / "EVENTS.jsonl").read_bytes())}
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            result = transcript.run_decoder_subprocess(batch, [entry], transcript.CPU_REFERENCE_PYTHON)
            self.assertEqual(result["artifacts"][0]["status"], "complete")
            document = yaml.safe_load(Path(result["artifacts"][0]["path"]).read_bytes())
            self.assertEqual(document["generations"][0]["reasoning_text"], "Synthetic recorded reasoning.\n")
            self.assertEqual(document["generations"][0]["action_text"], '\n{"action":"wait"}')
            self.assertEqual(document["tokenizer"]["manifest_sha256"], transcript.REFERENCE_MANIFEST_SHA256)
            self.assertEqual(document["tokenizer"]["defined_tokenizer_vocab_size"], 248077)
            self.assertEqual(document["tokenizer"]["native_model_vocab_size"], 248320)
            for path, raw in before.items():
                self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
