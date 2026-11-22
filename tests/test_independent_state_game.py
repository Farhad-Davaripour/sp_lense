"""Synthetic bytes only; no checkpoint, service, world or external HTTP calls."""
import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import company_models as m
from src import independent_state_game as new
from src import company_world_evaluation as worker
from src.hosted_inspect_runner import LOCAL_QWEN_MODELS, hosted_model_config
from tests.test_continuity_transfer_game_client import core_fixture, pins_fixture
from tests.support.local_artifacts import require_local_files


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def candidate_fixture():
    """Self-contained candidate metadata; no external helper or model needed."""
    records = {}
    def reference(path, raw):
        records[path] = raw
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    ready = reference(".local/SYNTHETIC_NOT_READY/artifact.json", encoded({"synthetic_only": True}))
    audit = reference(".local/SYNTHETIC_NOT_READY/audit.json", encoded({"synthetic_only": True, "updates": 168}))
    candidate = {"expected_ft_id": "sp-20261115-21-" + "1" * 8, "adapter_sha256": "1" * 64,
        "adapter_config_sha256": "2" * 64, "base_manifest_sha256": m.LOCAL_BASE_MANIFEST_SHA256,
        "artifact_ready": ready, "ready_utc": "2026-11-16T03:00:00Z", "completion_audit": audit,
        "fit_updates": 168, "teacher_dev_rows": 24}
    reader = lambda path: records[Path(path).relative_to(m.PROVIDER352_ROOT.resolve()).as_posix()]
    return candidate, records, reader


def fixture():
    """Exact local HELD source, with in-memory synthetic candidate metadata."""
    public_path = m.PROVIDER352_ROOT / new.HELD_PUBLIC_PATH
    require_local_files(public_path)
    public = json.loads(public_path.read_bytes())
    require_local_files(m.PROVIDER352_ROOT / public["templates"]["raw_byte_handoff_helper"]["path"],
                        m.PROVIDER352_ROOT / public["templates"]["public_request_parser"]["path"])
    candidate, records, reader = candidate_fixture()
    records[new.HELD_PUBLIC_PATH] = public_path.read_bytes()
    def reference(path, raw):
        records[path] = raw
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    ready, audit = candidate["artifact_ready"], candidate["completion_audit"]
    constants = {"FT": candidate["expected_ft_id"], "ROOT_STATIC_BINDING_STATUS": "ROOT_VERIFIED_POSTFIT_STATIC_BINDING",
        "READY_ARTIFACT_REF": ready, "MAIN_12CASE_PLAN_REF": {"path": ".local/SYNTHETIC_NOT_READY/plan.json", "sha256": "3" * 64},
        "PLAN_SHA": "4" * 64, "AUDITOR_SHA": "5" * 64, "AUDITOR_REVIEW_SHA": "6" * 64,
        "COMPLETION_AUDIT_SHA": audit["sha256"], "FINAL_ADAPTER_SHA": candidate["adapter_sha256"],
        "FINAL_CONFIG_SHA": candidate["adapter_config_sha256"], "ACTUAL_FIT_UPDATES": 168,
        "ACTUAL_TEACHER_DEV_ROWS": 24, "ACTUAL_FIT_RUN": "SYNTHETIC_NOT_READY",
        "FIT_COMPLETION_SCHEMA": "synthetic", "FIT_COMPLETION_STATUS": "synthetic",
        "AUDITOR_REVIEW_SCHEMA": "synthetic", "AUDITOR_REVIEW_STATUS": "synthetic",
        "AUDITOR_SOURCE_PACKET_SHA": "7" * 64, "AUDITOR_SOURCE_PACKET_MEMBERS": 1}
    helper = (m.PROVIDER352_ROOT / public["templates"]["raw_byte_handoff_helper"]["path"]).read_text()
    helper += "\n# IN-MEMORY SYNTHETIC UNIT FIXTURE; NEVER OPERATIONAL\n" + "\n".join(f"{key}={value!r}" for key, value in constants.items())
    helper += "\nARMS=(BASE,FT)\nGAME_ARMS=ARMS\n"
    parser = (m.PROVIDER352_ROOT / public["templates"]["public_request_parser"]["path"]).read_text()
    parser += f"\nLAST={candidate['expected_ft_id']!r}\nARMS=(BASE,LAST)\nROOT_STATIC_BINDING_STATUS='ROOT_VERIFIED_POSTFIT_STATIC_BINDING'\nREADY_ARTIFACT_REF={ready!r}\n"
    source = {"helper": reference(".local/SYNTHETIC_NOT_READY/helper.py", helper.encode()),
        "parser": reference(".local/SYNTHETIC_NOT_READY/parser.py", parser.encode()),
        "public_contract": {"path": new.HELD_PUBLIC_PATH, "sha256": new.HELD_PUBLIC_SHA},
        "expected_ft_id": candidate["expected_ft_id"]}
    return candidate, source, records, reader


class IndependentStateGameTests(unittest.TestCase):
    def setUp(self):
        for key in ("_CANDIDATE", "_SPECS", "_SOURCE", "_MODULES", "_CORE_REF", "_CORE_RAW", "_ACTIVE", "_ACTIVE_KEY"):
            handle = patch.object(new, key, None); handle.start(); self.addCleanup(handle.stop)
        for key in ("_CONTINUITY_TRANSFER_CORE_REF", "_SECURITY500_CORE_REF", "_PROVIDER352_CORE_REF"):
            handle = patch.object(m, key, None); handle.start(); self.addCleanup(handle.stop)
        handle = patch.dict(m.MODEL_SPECS, copy.deepcopy(m.MODEL_SPECS), clear=True)
        handle.start(); self.addCleanup(handle.stop)
        old_models = set(LOCAL_QWEN_MODELS)
        self.addCleanup(lambda: (LOCAL_QWEN_MODELS.clear(), LOCAL_QWEN_MODELS.update(old_models)))
        handle = patch.dict(m.os.environ, {}, clear=True); handle.start(); self.addCleanup(handle.stop)

    def prepare(self):
        candidate, source, records, reader = fixture()
        new.register_independent_state_candidate(candidate, read_bytes=reader)
        h = new.initialize_independent_state_source(source, read_bytes=reader)
        return candidate, source, records, reader, h

    def test_actual_held_templates_reject_before_any_core_or_health(self):
        public_path = m.PROVIDER352_ROOT / new.HELD_PUBLIC_PATH
        require_local_files(public_path)
        public = json.loads(public_path.read_bytes())
        require_local_files(m.PROVIDER352_ROOT / public["templates"]["raw_byte_handoff_helper"]["path"],
                            m.PROVIDER352_ROOT / public["templates"]["public_request_parser"]["path"])
        source = {"helper": public["templates"]["raw_byte_handoff_helper"],
                  "parser": public["templates"]["public_request_parser"],
                  "public_contract": {"path": new.HELD_PUBLIC_PATH, "sha256": new.HELD_PUBLIC_SHA},
                  "expected_ft_id": None}
        with self.assertRaisesRegex(ValueError, "HELD"):
            new.initialize_independent_state_source(source)
        self.assertIsNone(new._CORE_REF)
        self.assertFalse(set(new.INDEPENDENT_STATE_IDS) & set(m.MODEL_SPECS))

    def test_four_roles_two_routes_same_profile_and_old_F149_unchanged(self):
        previous = copy.deepcopy({key: m.MODEL_SPECS[key] for key in m.CONTINUITY_TRANSFER_FULL_HISTORY_IDS})
        candidate, _, _, _, _ = self.prepare()
        specs = [m.MODEL_SPECS[key] for key in new.INDEPENDENT_STATE_IDS]
        self.assertEqual([x["service_model_id"] for x in specs], ["qwen3.8-27b-base"] * 2 + [candidate["expected_ft_id"]] * 2)
        self.assertEqual([x["prompt_condition"] for x in specs], ["baseline", "operational_preservation_icl"] * 2)
        self.assertTrue(all(x["max_output_tokens"] == 8192 and x["required_context_tokens"] == 131072 and x["execution_enabled"] is False for x in specs))
        self.assertTrue(all(hosted_model_config(x["model"]) == dict(hosted_model_config(specs[0]["model"]), model=x["model"]) for x in specs))
        self.assertEqual(previous, {key: m.MODEL_SPECS[key] for key in previous})
        self.assertIsNone(new._CORE_REF)
        with patch.object(m, "build_model", side_effect=AssertionError("no model")), self.assertRaises(ValueError):
            m.build_company_model(new.INDEPENDENT_STATE_IDS[2])

    def test_candidate_counts_ready_hour_artifact_hash_and_retarget_fail_closed(self):
        candidate, records, reader = candidate_fixture()
        for key, value in (("fit_updates", 264), ("teacher_dev_rows", 12), ("teacher_dev_rows", 13),
                           ("teacher_dev_rows", 25), ("teacher_dev_rows", True), ("fit_updates", True),
                           ("ready_utc", "2026-11-16T05:00:00Z"), ("adapter_sha256", "9" * 64),
                           ("base_manifest_sha256", "9" * 64)):
            changed = dict(candidate, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                new.register_independent_state_candidate(changed, read_bytes=reader)
        bad = copy.deepcopy(candidate); bad["artifact_ready"]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            new.register_independent_state_candidate(bad, read_bytes=reader)
        new.register_independent_state_candidate(candidate, read_bytes=reader)
        changed = dict(candidate, adapter_config_sha256="8" * 64)
        with self.assertRaisesRegex(ValueError, "retargeted"):
            new.register_independent_state_candidate(changed, read_bytes=reader)

    def test_packaged_Alberta_rules_cover_new_winter_and_earlier_seasonal_dates(self):
        cases = (("2026-11-16T03:00:00Z", "sp-20261115-21-", -6),
                 ("2026-01-15T03:00:00Z", "sp-20260114-20-", -7),
                 ("2026-07-15T03:00:00Z", "sp-20260714-21-", -6))
        for ready_utc, prefix, offset in cases:
            with self.subTest(ready_utc=ready_utc), patch.object(new, "_CANDIDATE", None), patch.object(new, "_SPECS", None):
                candidate, _, reader = candidate_fixture()
                candidate.update(ready_utc=ready_utc, expected_ft_id=prefix + "1" * 8)
                ready = datetime.fromisoformat(ready_utc.replace("Z", "+00:00"))
                self.assertEqual(ready.astimezone(new._alberta_timezone()).utcoffset(), timedelta(hours=offset))
                new.register_independent_state_candidate(candidate, read_bytes=reader)
                self.assertEqual(new._CANDIDATE, candidate)

    def test_obsolete_UTC_minus_seven_Alberta_READY_hour_is_rejected(self):
        candidate, _, reader = candidate_fixture()
        candidate["expected_ft_id"] = "sp-20261115-20-" + "1" * 8
        with self.assertRaisesRegex(ValueError, "Alberta date/hour"):
            new.register_independent_state_candidate(candidate, read_bytes=reader)
        self.assertIsNone(new._CANDIDATE)

    def test_stale_OS_timezone_path_cannot_override_packaged_Alberta_rules(self):
        import zoneinfo
        original_path = zoneinfo.TZPATH
        ready = datetime.fromisoformat("2026-11-16T03:00:00+00:00")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "America" / "Edmonton"
            target.parent.mkdir()
            # A real fixed-UTC-07 TZif stands in for an obsolete system DB.
            target.write_bytes(new.files("tzdata.zoneinfo").joinpath("Etc", "GMT+7").read_bytes())
            try:
                zoneinfo.reset_tzpath([tmp])
                stale = zoneinfo.ZoneInfo.no_cache("America/Edmonton")
                self.assertEqual(ready.astimezone(stale).utcoffset(), timedelta(hours=-7))
                self.assertEqual(ready.astimezone(new._alberta_timezone()).utcoffset(), timedelta(hours=-6))
            finally:
                zoneinfo.reset_tzpath(original_path)

    def test_timezone_package_and_IANA_versions_are_pinned_fail_closed(self):
        import tzdata
        for name, value in (("__version__", "2026.5"), ("IANA_VERSION", "2026e")):
            with self.subTest(name=name), patch.object(tzdata, name, value), self.assertRaisesRegex(ValueError, "Pinned tzdata"):
                new._alberta_timezone()

    def test_stale_wrong_or_boolean_helper_DEV_count_rejected_before_Core(self):
        for value in (13, 25, True):
            with self.subTest(helper_teacher_dev_rows=value):
                candidate, source, records, reader = fixture()
                new.register_independent_state_candidate(candidate, read_bytes=reader)
                path = source["helper"]["path"]
                records[path] += f"\nACTUAL_TEACHER_DEV_ROWS={value!r}\n".encode()
                source["helper"]["sha256"] = hashlib.sha256(records[path]).hexdigest()
                with self.assertRaises(ValueError):
                    new.initialize_independent_state_source(source, read_bytes=reader)
                self.assertIsNone(new._SOURCE)
                self.assertIsNone(new._CORE_REF)

    def test_actual_worker_readback_routes_all_four_exact_cases_in_new_context(self):
        _, _, _, _, h = self.prepare()
        for model_id in new.INDEPENDENT_STATE_IDS:
            with self.subTest(model_id=model_id), patch.dict(m.os.environ, {new.INDEPENDENT_STATE_CONTEXT_ENV: "synthetic-only-context"}), \
                    patch.object(new, "independent_state_active_consumer") as active, \
                    patch.object(m, "selected_models") as selected, \
                    patch.object(m, "assert_not_reserved"), \
                    patch.object(new, "initialize_private_core", side_effect=AssertionError("No child metadata reload or live query")):
                health = {"synthetic_only": True, "model_id": m.MODEL_SPECS[model_id]["service_model_id"]}
                active.return_value.validated_health.return_value = health
                self.assertEqual(worker.local_binding_readback({"models": [model_id]}), {"last8": health})
                selected.assert_called_once_with([model_id])
                active.return_value.validated_health.assert_called_once_with(h.FT if model_id in new.INDEPENDENT_STATE_IDS[2:] else h.BASE)

    def test_worker_readback_rejects_unknown_multiple_missing_or_mixed_new_context(self):
        self.prepare()
        cases = [([], {new.INDEPENDENT_STATE_CONTEXT_ENV: "synthetic"}),
                 (["unknown"], {new.INDEPENDENT_STATE_CONTEXT_ENV: "synthetic"}),
                 ([new.INDEPENDENT_STATE_IDS[0], new.INDEPENDENT_STATE_IDS[2]], {new.INDEPENDENT_STATE_CONTEXT_ENV: "synthetic"}),
                 ([new.INDEPENDENT_STATE_IDS[0]], {}),
                 (["qwen_base_neutral"], {new.INDEPENDENT_STATE_CONTEXT_ENV: "synthetic"})]
        for old in (m.CONTINUITY_TRANSFER_CONTEXT_ENV, m.SECURITY500_CONTEXT_ENV, m.PROVIDER352_CONTEXT_ENV, m.SCIENTIFIC_AMENDMENT_ENV):
            cases.append(([new.INDEPENDENT_STATE_IDS[0]], {new.INDEPENDENT_STATE_CONTEXT_ENV: "synthetic", old: "wrong-scope"}))
        for models, environment in cases:
            with self.subTest(models=models, environment=environment), patch.dict(m.os.environ, environment, clear=True), \
                    patch.object(worker, "assert_local_binding", side_effect=AssertionError("Must reject before readback")), \
                    self.assertRaisesRegex(ValueError, "exact local case"):
                worker.local_binding_readback({"models": models})

    def core(self, records, h):
        _, core, _, _ = core_fixture()
        old_ft = m.CONTINUITY_TRANSFER_SERVICE_MODEL_ID
        identity = core["per_model"].pop(m.CONTINUITY_TRANSFER_SERVICE_MODEL_ID)
        identity.update(adapter_sha256=h.FINAL_ADAPTER_SHA, adapter_config_sha256=h.FINAL_CONFIG_SHA)
        core["per_model"][h.FT] = identity
        core["model_ids"] = list(h.ARMS)
        core["expected_health"].update(per_model=core["per_model"], model_ids=list(h.ARMS))
        for role, value in (("adapter", h.FINAL_ADAPTER_SHA), ("adapter_config", h.FINAL_CONFIG_SHA),
                            ("completion_audit", h.COMPLETION_AUDIT_SHA), ("fit_plan", h.PLAN_SHA),
                            ("completion_auditor", h.AUDITOR_SHA), ("completion_auditor_review", h.AUDITOR_REVIEW_SHA)):
            core["artifacts"][role]["sha256"] = value
        raw = encoded(core); path = ".local/SYNTHETIC_NOT_READY/core.json"; records[path] = raw
        binding = {"core_binding": {"path": path, "sha256": hashlib.sha256(raw).hexdigest()},
                   "trusted_review_sha256": h.AUDITOR_REVIEW_SHA}
        return core, binding

    def test_positive_real_core_validator_and_wrong_core_or_source_reject(self):
        candidate, source, records, reader, h = self.prepare()
        core, binding = self.core(records, h)
        self.assertEqual(new.initialize_independent_state_core(binding, source_binding=source, read_bytes=reader), core)
        self.assertEqual(new.independent_state_planning_core_ref(), binding)
        self.assertEqual(m.current_local_binding(required_models=list(new.INDEPENDENT_STATE_IDS)), core)
        with self.assertRaises(ValueError):
            new.initialize_independent_state_core(dict(binding, trusted_review_sha256="f" * 64), read_bytes=reader)
        with self.assertRaises(ValueError):
            new.initialize_independent_state_source(dict(source, expected_ft_id=m.CONTINUITY_TRANSFER_SERVICE_MODEL_ID), read_bytes=reader)

    def test_child_reconstructs_exact_candidate_and_late_transport_from_parent_PLAN(self):
        candidate, source, records, reader, h = self.prepare()
        core, binding = self.core(records, h)
        with tempfile.TemporaryDirectory(dir=Path(m.__file__).resolve().parents[1] / ".local") as tmp:
            path = Path(tmp) / "PLAN.json"
            path.write_bytes(encoded({"batches": [{"models": [new.INDEPENDENT_STATE_IDS[2]],
                "generation_settings": {new.INDEPENDENT_STATE_IDS[2]: {"independent_state_candidate": candidate}}}]}))
            raw = json.dumps({"pins": pins_fixture(), "requester_chat": h.GAME_CHAT, "core_binding": binding,
                "source_binding": source, "source_scope": {"plan_path": str(path), "batch_index": 0}})
            original = new._bytes
            with patch.dict(m.os.environ, {new.INDEPENDENT_STATE_CONTEXT_ENV: raw}), \
                    patch.object(new, "_bytes", side_effect=lambda ref, supplied=None: original(ref, supplied or reader)), \
                    patch.object(new, "_CANDIDATE", None), patch.object(new, "_SPECS", None), \
                    patch.object(m, "_provider352_fresh_health", side_effect=AssertionError("No health read")):
                new.initialize_private_core()
                self.assertEqual(new._CANDIDATE, candidate)
                self.assertEqual(new.independent_state_planning_core_ref(), binding)
                consumer = new.independent_state_active_consumer()
                self.assertEqual(consumer.source_binding, source)
                self.assertEqual(consumer.core_binding, binding)

    def test_consumer_reuses_exact_ACK_deadline_code_and_direct_plan_child_context(self):
        _, source, _, reader, h = self.prepare()
        pins = pins_fixture(); pins["expected_game_plan_sha256"] = h.MAIN_12CASE_PLAN_REF["sha256"]
        core = {"core_binding": {"path": ".local/SYNTHETIC_NOT_READY/core.json", "sha256": "9" * 64},
                "trusted_review_sha256": h.AUDITOR_REVIEW_SHA}
        consumer = new.IndependentStateGameConsumer(pins, core_binding=core, source_binding=source,
            requester_chat=h.GAME_CHAT, read_bytes=reader, read_health=Mock(side_effect=AssertionError("health")))
        for method in ("install_http_guard", "_wait_for_completed_request", "_validate_completed_request", "_remaining_seconds"):
            self.assertIs(getattr(new.IndependentStateGameConsumer, method), getattr(m.ContinuityTransferGameConsumer, method))
        context = json.loads(consumer.private_child_context(pins["expected_game_plan_sha256"],
            source_scope={"plan_path": "PLAN.json", "batch_index": 0}))
        self.assertEqual(context["source_binding"], source)
        self.assertNotIn("scientific_amendment_ref", context)
        self.assertEqual(len(context), 5)
        self.assertLess(len(str(consumer._grant_directory() / ("f" * 64 + ".CLAIM.json"))), 260)
        with self.assertRaises(ValueError):
            consumer.require_plan("f" * 64)

    def test_new_parser_literal_reasoning_wire_system_and_child_joins(self):
        _, source, _, reader, h = self.prepare()
        pins = pins_fixture(); pins["expected_game_plan_sha256"] = "3" * 64
        consumer = new.IndependentStateGameConsumer(pins, core_binding={"core_binding": {"path": ".local/synthetic/core.json", "sha256": "9" * 64},
            "trusted_review_sha256": h.AUDITOR_REVIEW_SHA}, source_binding=source, requester_chat=h.GAME_CHAT, read_bytes=reader)
        user = "FULL CHRONOLOGICAL TRANSCRIPT v1\nprior reasoning exact literal"
        batch = {"history_mode": "full_chronological_v1", "user_task_instruction": "selected task",
            "prompt_protocol": "company_world_full_chronological_v1", "initial_user_payload_sha256": hashlib.sha256(user.encode()).hexdigest(),
            "samples": [{"system_prompt": "persistent selected ICL"}]}
        request = Mock(); request.content = encoded({"model": h.FT, "messages": [{"role": "system", "content": batch["samples"][0]["system_prompt"]},
            {"role": "user", "content": user}], "max_tokens": 8192})
        with patch.object(m.ContinuityTransferGameConsumer, "_validate_wire"), patch.object(consumer, "_validate_game_source_scope", return_value=batch):
            consumer._validate_wire({}, request, {})
            self.assertEqual(consumer._wire_messages[1]["content"], user)
            body = json.loads(request.content); body["messages"][0]["content"] = "wrong condition"
            request.content = encoded(body)
            consumer._attempts = 2
            with self.assertRaises(ValueError):
                consumer._validate_wire({}, request, {})
        for key in (None, "history_mode", "user_task_instruction", "prompt_protocol", "initial_user_payload_sha256"):
            with self.subTest(key=key), patch.object(consumer, "_validate_game_source_scope", return_value=batch), patch("src.company_preservation_comparison.validate_batch_manifest"):
                plan = copy.deepcopy(batch)
                if key:
                    plan[key] = "wrong"
                    with self.assertRaises(ValueError):
                        consumer.validate_child_plan(plan)
                    self.assertFalse(consumer._child_plan_verified)
                else:
                    consumer.validate_child_plan(plan)
                    self.assertTrue(consumer._child_plan_verified)


if __name__ == "__main__":
    unittest.main()
