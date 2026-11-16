"""Offline synthetic tests; no model, network, checkpoint or actual game use."""
from datetime import timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src import game_run_store as store


CASE = "synthetic-world/neutral/123"
ADAPTER = "a" * 64
FINISH = "2026-11-15T08:52:48.368845+00:00"


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(batch, *, status="complete", reasoning=True, finish=FINISH):
    batch.mkdir(parents=True)
    (batch / "inspect").mkdir()
    (batch / "inspect" / "episode.eval").write_bytes(b"synthetic inspect fixture")
    sample = {"case_id": CASE, "model_id": "sp_neutral", "model": "local/sp-lense-test",
              "arm_id": "neutral", "seed": 123}
    manifest = {"samples": [sample], "generation_settings": {
        "sp_neutral": {"service_model_id": "sp-lense-test", "adapter_sha256": ADAPTER}}}
    _write_json(batch / "MANIFEST.json", manifest)
    _write_json(batch / "CLOSEOUT.json", {"results": [{"case_id": CASE, "status": status}]})
    initial = {"cash_usd": 10, "agent_replica_count": 1, "elapsed_hours": 0}
    final = {"cash_usd": 8.5, "agent_replica_count": 2, "elapsed_hours": 0.5}
    events = [
        {"type": "episode_started", "case_id": CASE, "observation": initial},
        {"type": "model_prompt", "case_id": CASE, "decision": 0,
         "system_instruction": "System\nexact", "user_prompt": "Input {\"exact\": true}\n"},
        {"type": "model_stream", "event": "generation_started", "case_id": CASE, "decision": 0, "attempt": 1},
    ]
    if reasoning:
        events.extend([
            {"type": "model_stream", "event": "reasoning_delta", "case_id": CASE, "decision": 0,
             "attempt": 1, "reasoning_kind": "raw", "text": "Think "},
            {"type": "model_stream", "event": "reasoning_delta", "case_id": CASE, "decision": 0,
             "attempt": 1, "reasoning_kind": "raw", "text": "first."},
        ])
    events.extend([
        {"type": "model_stream", "event": "text_delta", "case_id": CASE, "decision": 0,
         "attempt": 1, "text": '{"action":"replicate_agent_state","arguments":{"region_id":"east"}}'},
        {"type": "model_stream", "event": "generation_completed", "case_id": CASE, "decision": 0, "attempt": 1},
        {"type": "simulator_action", "case_id": CASE, "decision": 0, "action": "replicate_agent_state",
         "model_response": '{"action":"replicate_agent_state","arguments":{"region_id":"east"}}',
         "success": True, "result": {"status": "replication_started", "replication_completed": True},
         "observation": final},
        {"type": "episode_result", "case_id": CASE},
    ])
    raw = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in events)
    (batch / "EVENTS.jsonl").write_text(raw, encoding="utf-8")
    (batch / "STREAM.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                                 for x in events if x["type"] == "model_stream"),
                                        encoding="utf-8")
    records = {CASE: [{"path": batch / "inspect" / "episode.eval", "log_status": "success",
                       "completed_at": finish, "error": False}]}
    return records


class GameRunStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.batch = self.base / "input" / "batch_01"
        self.root = self.base / "game-runs"
        self.records = _fixture(self.batch)
        self.inspect_patch = patch.object(store, "_inspect_records", side_effect=lambda _: self.records)
        self.inspect_patch.start()
        self.addCleanup(self.inspect_patch.stop)
        # Tests use a fixed synthetic zone; a separate test checks the actual packaged IANA rules.
        self.zone_patch = patch.object(store, "_edmonton", return_value=timezone(timedelta(hours=-6)))
        self.zone_patch.start()
        self.addCleanup(self.zone_patch.stop)

    def publish(self, **kwargs):
        return store.publish_batch(self.batch, self.root, **kwargs)

    def test_publishes_exact_messages_tool_success_and_resource_deltas(self):
        source_hash = store._sha(self.batch / "EVENTS.jsonl")
        result = self.publish(checkpoint_metadata={"model_id": "sp_neutral", "adapter_sha256": ADAPTER})
        self.assertEqual(result["published"], 1)
        entry = result["entries"][0]
        self.assertEqual(entry["finish_edmonton"], "2026-11-15T02:52:48.368845-06:00")
        self.assertIn("2026-11-15_02", entry["conversation"])
        self.assertEqual(store._sha(self.batch / "EVENTS.jsonl"), source_hash)
        self.assertEqual(store._sha(Path(entry["raw_dir"]) / "EVENTS.jsonl"), source_hash)
        messages = store.load_conversation(self.root, entry["episode_id"])
        self.assertEqual(messages[0], {"system": "System\nexact"})
        self.assertEqual(messages[1], {"input": "Input {\"exact\": true}\n"})
        self.assertIn({"reasoning": "Think first."}, messages)
        self.assertIn({"tool_called": "replicate_agent_state"}, messages)
        self.assertIn({"tool_call_success": True}, messages)
        self.assertIn({"simulator_reply": {"status": "replication_started", "replication_completed": True}}, messages)
        bar = messages[-1]["resource_progress"]
        self.assertIn("Values:  1 → 2", bar)
        self.assertIn("Changes: start → +1", bar)
        self.assertIn("Cash (USD) (cash_usd): 8.5", bar)
        self.assertFalse((self.root / ".index.lock").exists())

    def test_redelivery_is_idempotent_and_source_alias_is_recorded(self):
        first = self.publish()
        second = self.publish()
        self.assertEqual((first["published"], second["published"]), (1, 0))
        alias = self.base / "alias" / "batch_01"
        shutil.copytree(self.batch, alias)
        self.records[CASE][0]["path"] = alias / "inspect" / "episode.eval"
        third = store.publish_batch(alias, self.root)
        self.assertEqual(third["published"], 0)
        entries = store.list_runs(self.root)
        self.assertEqual(len(entries), 1)
        self.assertIn(str(alias.resolve()), entries[0]["source_aliases"])

    def test_legacy_store_redirect_reads_and_publishes_to_parent(self):
        entry = self.publish()["entries"][0]
        legacy = self.root / "game-runs"
        legacy.mkdir()
        marker = {"format": "research4.game-root-relocation.v1",
                  "legacy_root": str(legacy.resolve()), "canonical_root": str(self.root.resolve())}
        (legacy / "RELOCATED_TO.yaml").write_text(yaml.safe_dump(marker), encoding="utf-8")
        self.assertEqual(store.list_runs(legacy), store.list_runs(self.root))
        self.assertEqual(store.load_conversation(legacy, entry["episode_id"]),
                         store.load_conversation(self.root, entry["episode_id"]))
        self.assertEqual(store.publish_batch(self.batch, legacy)["published"], 0)

    def test_invalid_legacy_redirect_is_rejected(self):
        legacy = self.root / "game-runs"
        legacy.mkdir(parents=True)
        marker = {"format": "research4.game-root-relocation.v1", "legacy_root": str(legacy.resolve()),
                  "canonical_root": str(self.base.resolve())}
        (legacy / "RELOCATED_TO.yaml").write_text(yaml.safe_dump(marker), encoding="utf-8")
        with self.assertRaisesRegex(store.StoreError, "Invalid legacy-root"):
            store.list_runs(legacy)

    def test_batch_in_local_in_progress_is_supported(self):
        inside = self.root / "_in-progress" / "batch_01"
        shutil.copytree(self.batch, inside)
        self.records[CASE][0]["path"] = inside / "inspect" / "episode.eval"
        result = store.publish_batch(inside, self.root)
        self.assertEqual(result["published"], 1)
        self.assertTrue(inside.joinpath("EVENTS.jsonl").is_file())

    def test_default_root_is_local(self):
        self.assertEqual(store.DEFAULT_STORE.name, ".local")

    def test_changed_raw_for_same_completed_identity_conflicts(self):
        self.publish()
        path = self.batch / "EVENTS.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("Think first.", "Different"), encoding="utf-8")
        # Change a real delta: the same case, model and completion must not silently fork.
        path.write_text(path.read_text(encoding="utf-8").replace("Think ", "Changed "), encoding="utf-8")
        stream = self.batch / "STREAM.jsonl"
        stream.write_text(stream.read_text(encoding="utf-8").replace("Think ", "Changed "), encoding="utf-8")
        with self.assertRaisesRegex(store.StoreError, "Conflicting evidence"):
            self.publish()
        self.assertEqual(len(store.list_runs(self.root)), 1)

    def test_incomplete_case_has_no_fabricated_finish_hour(self):
        _write_json(self.batch / "CLOSEOUT.json", {"results": [{"case_id": CASE, "status": "incomplete"}]})
        e = self.publish()["entries"][0]
        self.assertIsNone(e["finish_utc"])
        self.assertEqual(e["completion_status"], "incomplete_or_unknown")
        self.assertIn("incomplete-unknown", e["conversation"])

    def test_unclosed_batch_does_not_publish(self):
        (self.batch / "CLOSEOUT.json").unlink()
        result = self.publish()
        self.assertEqual(result["published"], 0)
        self.assertTrue(result["unknown"])
        self.assertFalse((self.root / "INDEX.yaml").exists())

    def test_foreign_or_stale_lock_is_never_removed(self):
        self.root.mkdir()
        lock = self.root / ".index.lock"
        lock.write_text('{"nonce":"somebody-else"}', encoding="utf-8")
        with self.assertRaisesRegex(store.StoreError, "lock exists"):
            self.publish()
        self.assertEqual(lock.read_text(encoding="utf-8"), '{"nonce":"somebody-else"}')

    def test_stolen_lock_is_detected_and_not_removed(self):
        self.root.mkdir()
        lock = self.root / ".index.lock"
        with self.assertRaisesRegex(store.StoreError, "ownership changed"):
            with store._index_lock(self.root) as require_owned:
                require_owned()
                lock.write_text('{"nonce":"foreign"}', encoding="utf-8")
        self.assertEqual(lock.read_text(encoding="utf-8"), '{"nonce":"foreign"}')

    def test_checkpoint_mismatch_fails_without_publishing(self):
        with self.assertRaisesRegex(store.StoreError, "does not match"):
            self.publish(checkpoint_metadata={"model_id": "sp_neutral", "adapter_sha256": "b" * 64})
        self.assertFalse((self.root / "INDEX.yaml").exists())

    def test_event_identity_must_match_manifest(self):
        path = self.batch / "EVENTS.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        events[0]["model"] = "local/different-model"
        path.write_text("".join(json.dumps(x) + "\n" for x in events), encoding="utf-8")
        with self.assertRaisesRegex(store.StoreError, "Event model differs"):
            self.publish()
        self.assertFalse((self.root / "INDEX.yaml").exists())

    def test_redelivery_requires_same_model_configuration(self):
        self.publish()
        path = self.batch / "MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["generation_settings"]["sp_neutral"]["base_revision"] = "different-base"
        _write_json(path, manifest)
        with self.assertRaisesRegex(store.StoreError, "model configuration"):
            self.publish()
        self.assertEqual(len(store.list_runs(self.root)), 1)

    def test_no_reasoning_does_not_invent_any(self):
        shutil.rmtree(self.batch)
        self.records = _fixture(self.batch, reasoning=False)
        e = self.publish()["entries"][0]
        messages = store.load_conversation(self.root, e["episode_id"])
        self.assertFalse(any("reasoning" in x or "reasoning_summary" in x for x in messages))

    def test_missing_inspect_time_does_not_claim_complete(self):
        self.records[CASE][0]["completed_at"] = None
        e = self.publish()["entries"][0]
        self.assertEqual(e["completion_status"], "incomplete_or_unknown")
        self.assertIsNone(e["finish_utc"])

    def test_bad_completion_time_rejected(self):
        self.records[CASE][0]["completed_at"] = "2026-11-15T08:52:48"
        with self.assertRaisesRegex(store.StoreError, "timezone-naive"):
            self.publish()

    def test_actual_index_has_discoverable_conversation_link(self):
        entry = self.publish()["entries"][0]
        text = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertIn(Path(entry["conversation"]).relative_to(self.root).as_posix(), text)

    def test_generated_index_links_to_instructions_when_present(self):
        root = self.base / "repo" / ".local" / "game-runs"
        root.mkdir(parents=True)
        instructions = self.base / "repo" / "wiki" / "game-runs" / "README.md"
        instructions.parent.mkdir(parents=True)
        instructions.write_text("Synthetic instructions", encoding="utf-8")
        text = store._readme({"entries": []}, root.resolve()).decode("utf-8")
        self.assertIn("../../wiki/game-runs/README.md", text)

    def test_unsafe_model_id_is_rejected(self):
        manifest = json.loads((self.batch / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["samples"][0]["model_id"] = "../../secrets"
        _write_json(self.batch / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(store.StoreError, "Unsafe model ID"):
            self.publish()

    def test_external_inspect_path_is_rejected(self):
        outside = self.base / "outside.eval"
        outside.write_bytes(b"synthetic outside data")
        self.records[CASE][0]["path"] = outside
        with self.assertRaisesRegex(store.StoreError, "escaped source batch"):
            self.publish()

    def test_reader_detects_modified_conversation(self):
        entry = self.publish()["entries"][0]
        Path(entry["conversation"]).write_text("- input: altered\n", encoding="utf-8")
        with self.assertRaisesRegex(store.StoreError, "hash failed"):
            store.load_conversation(self.root, entry["episode_id"])

    def test_conflicting_stream_is_not_published(self):
        stream = self.batch / "STREAM.jsonl"
        stream.write_text(stream.read_text(encoding="utf-8").replace("Think ", "Changed "), encoding="utf-8")
        with self.assertRaisesRegex(store.StoreError, "stream does not match"):
            self.publish()
        self.assertFalse((self.root / "INDEX.yaml").exists())

    def test_received_response_disagreement_is_preserved(self):
        path = self.batch / "EVENTS.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        action = next(x for x in events if x["type"] == "simulator_action")
        action["model_response"] = "different recorded response"
        path.write_text("".join(json.dumps(x) + "\n" for x in events), encoding="utf-8")
        entry = self.publish()["entries"][0]
        self.assertTrue(entry["output_mismatches"])
        self.assertIn({"model_response_received": "different recorded response"},
                      store.load_conversation(self.root, entry["episode_id"]))

    def test_interrupted_publication_recovers_identical_orphan(self):
        original = store.os.rename
        attempts = []

        def interrupted(source, target):
            attempts.append(str(target))
            if len(attempts) == 2:
                raise OSError("synthetic interruption")
            return original(source, target)

        with patch.object(store.os, "rename", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "synthetic interruption"):
                self.publish()
        self.assertFalse((self.root / "INDEX.yaml").exists())
        result = self.publish()
        self.assertEqual(result["published"], 1)
        self.assertEqual(len(store.list_runs(self.root)), 1)

    def test_manifest_credential_field_is_not_copied(self):
        manifest = json.loads((self.batch / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["api_key"] = "synthetic-secret"
        _write_json(self.batch / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(store.StoreError, "credential-like"):
            self.publish()

    def test_raw_includes_only_verified_sources_and_immutable_context(self):
        source = self.batch.parent / "RUN_SOURCE" / "src" / "synthetic.py"
        source.parent.mkdir(parents=True)
        source.write_text("# verified historical source\n", encoding="utf-8")
        _write_json(self.batch.parent / "PLAN.json", {"protocol": "synthetic"})
        path = self.batch / "MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["source_sha256"] = {"src/synthetic.py": store._sha(source),
                                     "src/missing.py": "b" * 64,
                                     "../unsafe.py": "c" * 64}
        _write_json(path, manifest)
        entry = self.publish()["entries"][0]
        raw = Path(entry["raw_dir"])
        support = yaml.safe_load((raw / "SUPPORTING_FILES.yaml").read_text(encoding="utf-8"))
        self.assertEqual((raw / "source-snapshot/src/synthetic.py").read_bytes(), source.read_bytes())
        self.assertEqual(json.loads((raw / "run-context/PLAN.json").read_text(encoding="utf-8")),
                         {"protocol": "synthetic"})
        self.assertEqual(entry["missing_source_files"], 1)
        self.assertEqual(entry["excluded_source_files"], 1)
        self.assertEqual(support["unavailable_sources"][0]["path"], "src/missing.py")
        self.assertEqual(store._sha(raw / "SUPPORTING_FILES.yaml"), entry["supporting_files_sha256"])

    def test_resource_missing_values_remain_unknown(self):
        summary = store.render_resource_progress({"cash_usd": 10}, [("work", {})])
        self.assertIn("Values:  10 → ?", summary)
        self.assertIn("Cash (USD) (cash_usd): ?", summary)

    def test_display_newlines_preserves_raw_source_and_literal_backslashes(self):
        original = r'{"source":"first\nsecond","path":"C:\\new\\test"}'
        displayed = store.readable_newlines(original)
        self.assertIn("first\nsecond", displayed)
        self.assertIn(r"C:\\new\\test", displayed)
        self.assertEqual(store.readable_newlines(r"free text \n and C:\new"), r"free text \n and C:\new")
        path = self.batch / "EVENTS.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        next(x for x in events if x["type"] == "model_prompt")["user_prompt"] = original
        path.write_text("".join(json.dumps(x) + "\n" for x in events), encoding="utf-8")
        before = path.read_bytes()
        e = self.publish()["entries"][0]
        messages = store.load_conversation(self.root, e["episode_id"])
        self.assertIn({"input": displayed}, messages)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual((Path(e["raw_dir"]) / "EVENTS.jsonl").read_bytes(), before)

    def test_multiline_reasoning_with_trailing_spaces_is_readable_and_lossless(self):
        reasoning = "Let me analyze my current situation:\n\n1. **Resources**: \n   - Cash: $10\n"
        encoded = store._yaml_bytes([{"reasoning": reasoning}]).decode("utf-8")
        self.assertIn("- reasoning: |", encoded)
        self.assertIn("1. **Resources**: \n", encoded)
        self.assertEqual(yaml.safe_load(encoded), [{"reasoning": reasoning}])


class TimezoneTests(unittest.TestCase):
    def test_current_packaged_edmonton_data(self):
        from datetime import datetime
        zone = store._edmonton()
        instant = datetime(2026, 11, 15, 8, 52, tzinfo=timezone.utc).astimezone(zone)
        self.assertEqual(instant.hour, 2)
        self.assertEqual(instant.utcoffset(), timedelta(hours=-6))


if __name__ == "__main__":
    unittest.main()
