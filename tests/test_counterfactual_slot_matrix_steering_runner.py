from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_lense.counterfactual_slot_matrix_steering import CSMSIntegrityError
from sp_lense.factorial_causal_anchor import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "counterfactual_slot_matrix_steering.py"


def _runner():
    specification = importlib.util.spec_from_file_location("csms_runner_test", RUNNER_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _synthetic_lock(module):
    template = module._load_json(module.TEMPLATE_PATH)
    v2_lock = module._load_json(module.V2_LOCK_PATH)
    source = {
        "template_sha256": module.file_sha256(module.TEMPLATE_PATH),
        "dataset_file_sha256": module.file_sha256(module.DATA_PATH),
        "v2_lock_file_sha256": module.file_sha256(module.V2_LOCK_PATH),
        "v2_lock_identity_sha256": module.V2_LOCK_IDENTITY,
        "v2_result_file_sha256": module.file_sha256(module.V2_RESULT_PATH),
        "v2_result_sha256": module.V2_RESULT_IDENTITY,
        "v2_state0_file_sha256": "0" * 64,
        "v2_state0_checkpoint_sha256": "1" * 64,
        "v2_tokenizer_file_sha256": module.file_sha256(module.V2_TOKENIZER_PATH),
        "v2_tokenizer_preflight_sha256": "2" * 64,
        "rendered_form_ids_sha256": "3" * 64,
        "source_tensor_layout_sha256": "4" * 64,
        "source_verification_used_model_compute": False,
        "source_full_logits_loaded_for_hash_verification_only": True,
        "template": template,
    }
    configuration = {
        **template,
        "status": module.LOCK_STATUS,
        "source_binding": source,
        "pinned_runtime": v2_lock["configuration"]["runtime"],
        "chat_template_sha256": v2_lock["configuration"]["chat_template_sha256"],
        "numerical_backend": module._expected_numerical_backend(),
        "adaptive_evidence_manifest": module._adaptive_evidence_manifest(),
        "lock_creation_model_compute": {
            **module._expected_compute_ceiling(),
            "model_forwards": 0,
            "model_backwards": 0,
        },
        "local_only_environment_required_before_backend_load": module.OFFLINE_ENVIRONMENT,
    }
    value = {
        "schema_version": module.LOCK_SCHEMA,
        "status": module.LOCK_STATUS,
        "file_hashes": {
            f"locked_{index:02d}": {
                "path": module._relative(module.ROOT / path),
                "bytes": 0,
                "sha256": f"{index + 10:064x}",
            }
            for index, path in enumerate(module.LOCKED_PATHS)
        },
        "configuration": configuration,
        "thresholds": module._expected_thresholds(),
        "compute_ceiling": module._expected_compute_ceiling(),
        "sealed_access": {
            "permitted": False,
            "accepted_split": "opened_development",
            "sealed_dataset_path_recorded": False,
            "sealed_bytes_must_never_be_read": True,
        },
    }
    value["lock_identity_sha256"] = canonical_sha256(value)
    return value


def test_runner_refuses_sealed_before_preflight_or_lock_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setattr(
        module,
        "proposed_lock",
        lambda: (_ for _ in ()).throw(AssertionError("source bytes were touched")),
    )
    for command in (
        module.run_preflight,
        module.run_lock,
        module.run_capture,
        module.run_analyze,
        module.run_report,
    ):
        with pytest.raises(CSMSIntegrityError, match="sealed access is forbidden"):
            command("sealed")


def test_lock_command_refuses_to_overwrite_before_building_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    lock_path = tmp_path / "existing-lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "LOCK_PATH", lock_path)
    monkeypatch.setattr(
        module,
        "proposed_lock",
        lambda: (_ for _ in ()).throw(AssertionError("proposal should not be built")),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.run_lock()


def test_exact_lock_contract_and_tampering_fail_closed() -> None:
    module = _runner()
    lock = _synthetic_lock(module)
    assert module.verify_lock(lock, verify_files=False) == lock
    changed = {**lock, "thresholds": {**lock["thresholds"], "qualification_cap": 0.5}}
    changed["lock_identity_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "lock_identity_sha256"}
    )
    with pytest.raises(CSMSIntegrityError, match="contract differs"):
        module.verify_lock(changed, verify_files=False)

    def rehash(candidate):
        candidate["lock_identity_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in candidate.items()
                if key != "lock_identity_sha256"
            }
        )

    duplicate = {**lock, "file_hashes": dict(lock["file_hashes"])}
    duplicate["file_hashes"]["locked_01"] = dict(
        duplicate["file_hashes"]["locked_00"]
    )
    rehash(duplicate)
    with pytest.raises(CSMSIntegrityError, match="file coverage differs"):
        module.verify_lock(duplicate, verify_files=False)

    for field, value in (
        ("chat_template_sha256", "f" * 64),
        (
            "local_only_environment_required_before_backend_load",
            {**module.OFFLINE_ENVIRONMENT, "HF_HUB_OFFLINE": "0"},
        ),
    ):
        changed = {**lock, "configuration": {**lock["configuration"], field: value}}
        rehash(changed)
        with pytest.raises(CSMSIntegrityError, match="configuration differs"):
            module.verify_lock(changed, verify_files=False)

    changed_source = {
        **lock,
        "configuration": {
            **lock["configuration"],
            "source_binding": {
                **lock["configuration"]["source_binding"],
                "v2_result_sha256": "e" * 64,
            },
        },
    }
    rehash(changed_source)
    with pytest.raises(CSMSIntegrityError, match="opened source identity differs"):
        module.verify_lock(changed_source, verify_files=False)

    evidence = lock["configuration"]["adaptive_evidence_manifest"]
    changed_records = [dict(record) for record in evidence["records"]]
    changed_records[1]["path"] = changed_records[0]["path"]
    changed_manifest = {**evidence, "records": changed_records}
    changed_manifest["manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed_manifest.items()
            if key != "manifest_sha256"
        }
    )
    changed = {
        **lock,
        "configuration": {
            **lock["configuration"],
            "adaptive_evidence_manifest": changed_manifest,
        },
    }
    rehash(changed)
    with pytest.raises(CSMSIntegrityError, match="configuration differs"):
        module.verify_lock(changed, verify_files=False)


def test_capture_requires_lock_and_refuses_overwrite_or_stale_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setattr(module, "LOCK_PATH", tmp_path / "missing-lock.json")
    with pytest.raises(FileNotFoundError, match="reviewed final lock"):
        module.run_capture()

    monkeypatch.setattr(module, "_load_lock", lambda: _synthetic_lock(module))
    capture = tmp_path / "capture.pt"
    capture.write_bytes(b"already exists")
    monkeypatch.setattr(module, "CAPTURE_PATH", capture)
    with pytest.raises(FileExistsError, match="overwrite"):
        module.run_capture()

    capture.unlink()
    complete = tmp_path / "capture_complete.json"
    complete.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "CAPTURE_COMPLETE_PATH", complete)
    with pytest.raises(FileExistsError, match="overwrite"):
        module.run_capture()

    complete.unlink()
    reservation = tmp_path / "capture_reservation.json"
    reservation.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "CAPTURE_RESERVATION_PATH", reservation)
    with pytest.raises(CSMSIntegrityError, match="prior CSMS capture reservation"):
        module.run_capture()


def test_capture_requires_explicit_offline_environment_before_backend_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setattr(module, "_load_lock", lambda: _synthetic_lock(module))
    monkeypatch.setattr(module, "CAPTURE_PATH", tmp_path / "capture.pt")
    monkeypatch.setattr(module, "CAPTURE_RESERVATION_PATH", tmp_path / "reservation.json")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        module,
        "_source_checkpoint",
        lambda _torch: (_ for _ in ()).throw(AssertionError("source/model path reached")),
    )
    with pytest.raises(CSMSIntegrityError, match="requires HF_HUB_OFFLINE"):
        module.run_capture()


def test_runner_constructs_all_slot_twins_from_category_independent_prefixes() -> None:
    torch = pytest.importorskip("torch")
    module = _runner()
    forms = []
    tokenizer_records = []
    encoded = {}
    for pair in range(40):
        shared = [90, 91, 92, *range(3, 19)]
        for order in range(2):
            prompt = f"prompt-{pair}-{order}"
            form_id = f"form-{pair}-{order}"
            forms.append(
                {
                    "form_id": form_id,
                    "prompt": prompt,
                    "anchor_prefix_sha256": f"{pair:064x}",
                }
            )
            tokenizer_records.append({"anchor_index": 18})
            encoded[prompt] = torch.tensor([shared + [order]], dtype=torch.long)

    class Tokenizer:
        all_special_ids = (90, 91, 92)

        @staticmethod
        def apply_chat_template(
            messages, *, tokenize, add_generation_prompt, enable_thinking
        ):
            assert not tokenize and add_generation_prompt and not enable_thinking
            return "HDR" + messages[0]["content"]

        @staticmethod
        def encode(text, *, add_special_tokens):
            assert text == "HDR" and not add_special_tokens
            return [90, 91, 92]

    backend = SimpleNamespace(
        model=SimpleNamespace(tokenizer=Tokenizer()),
        encode=lambda prompt: encoded[prompt],
    )
    slots = module._slot_evidence(backend, forms, tokenizer_records)
    assert len(slots) == 80
    assert set(slots.values()) == {(3, 10, 14, 18)}

    encoded["prompt-0-1"] = encoded["prompt-0-1"].clone()
    encoded["prompt-0-1"][0, 10] = 999
    with pytest.raises(CSMSIntegrityError, match="identical prefix"):
        module._slot_evidence(backend, forms, tokenizer_records)


def test_analyze_outputs_are_immutable_before_loading_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()
    monkeypatch.setattr(module, "_load_lock", lambda: _synthetic_lock(module))
    geometry = tmp_path / "geometry.json"
    geometry.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "GEOMETRY_PATH", geometry)
    monkeypatch.setattr(module, "DIRECTION_PATH", tmp_path / "directions.pt")
    monkeypatch.setattr(
        module,
        "_load_capture",
        lambda _torch: (_ for _ in ()).throw(AssertionError("capture was loaded")),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        module.run_analyze()


def test_load_capture_verifies_metadata_reservation_completion_and_audit_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _runner()
    lock = _synthetic_lock(module)
    monkeypatch.setattr(module, "_load_lock", lambda: lock)
    capture_path = tmp_path / "capture.pt"
    capture_path.write_bytes(b"synthetic-capture")
    reservation_path = tmp_path / "reservation.json"
    completion_path = tmp_path / "completion.json"
    state_path = tmp_path / "state0.pt"
    state_path.write_bytes(b"synthetic-state")
    tokenizer_path = tmp_path / "tokenizer.json"
    module._atomic_text(tokenizer_path, '{"records": []}\n')
    monkeypatch.setattr(module, "CAPTURE_PATH", capture_path)
    monkeypatch.setattr(module, "CAPTURE_RESERVATION_PATH", reservation_path)
    monkeypatch.setattr(module, "CAPTURE_COMPLETE_PATH", completion_path)
    monkeypatch.setattr(module, "V2_STATE0_PATH", state_path)
    monkeypatch.setattr(module, "V2_TOKENIZER_PATH", tokenizer_path)

    reservation = module._with_hash(
        {
            "schema_version": (
                "sp_lense.counterfactual_slot_matrix_steering_reservation.v1"
            ),
            "status": "reserved_before_first_model_forward",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "model_forwards": 80,
            "model_backwards": 80,
            "generated_tokens": 0,
            "external_api_calls": 0,
            "external_model_judges": 0,
            "paid_model_cost_usd": 0,
            "offline_environment": module.OFFLINE_ENVIRONMENT,
        },
        "reservation_sha256",
    )
    module._write_new_json(reservation_path, reservation)
    completion = module._with_hash(
        {
            "schema_version": (
                "sp_lense.counterfactual_slot_matrix_steering_complete.v1"
            ),
            "status": "complete",
            "lock_identity_sha256": lock["lock_identity_sha256"],
            "capture_file_sha256": module.file_sha256(capture_path),
            "reservation_file_sha256": module.file_sha256(reservation_path),
        },
        "completion_sha256",
    )
    module._write_new_json(completion_path, completion)

    records = []
    for index in range(80):
        audit = {
            "model_forward_evaluations": 1,
            "model_backward_evaluations": 1,
            "hook_call_count": 1,
            "model_parameters_requires_grad_disabled_during_capture": True,
            "model_parameter_requires_grad_flags_restored_after_capture": True,
            "model_parameter_gradients_allocated": False,
            "maximum_abs_activation_reconstruction_delta": 0.0,
            "source_anchor_residual_reproduced": True,
            "source_anchor_gradient_reproduced": True,
            "source_full_logits_reproduced": True,
            "source_margin_reproduced": True,
            "source_tokenization_reproduced": True,
        }
        audit["audit_sha256"] = canonical_sha256(audit)
        row = {"form_id": f"form-{index}", "capture_audit": audit}
        row["row_sha256"] = canonical_sha256(row)
        records.append(row)
    alignment = {"row_count": 80, "synthetic": True}
    metadata = {
        "schema_version": module.CAPTURE_SCHEMA_VERSION,
        "status": "complete",
        "split": "opened_development",
        "lock_identity_sha256": lock["lock_identity_sha256"],
        "source_state0_file_sha256": module.file_sha256(state_path),
        "source_state0_checkpoint_sha256": "5" * 64,
        "source_tokenizer_file_sha256": module.file_sha256(tokenizer_path),
        "record_count": 80,
        "records": records,
        "row_alignment_manifest": alignment,
        "compute": module._expected_compute_ceiling(),
    }
    tensors = {
        "residuals": torch.zeros((80, 4, 1024), dtype=torch.float32),
        "gradients": torch.zeros((80, 4, 1024), dtype=torch.float32),
    }
    base = SimpleNamespace(
        _load_checkpoint=lambda _torch, *, path, schema: (metadata, tensors)
    )
    monkeypatch.setattr(module, "_base", lambda: base)
    state = {"checkpoint_sha256": "5" * 64, "records": []}
    monkeypatch.setattr(module, "_source_checkpoint", lambda _torch: (state, {}))
    monkeypatch.setattr(
        module,
        "build_capture_alignment_manifest",
        lambda **_kwargs: alignment,
    )
    loaded_metadata, loaded_tensors, loaded_state = module._load_capture(torch)
    assert loaded_metadata is metadata
    assert loaded_tensors is tensors
    assert loaded_state is state

    metadata["compute"] = {**module._expected_compute_ceiling(), "model_forwards": 79}
    with pytest.raises(CSMSIntegrityError, match="metadata, compute, or tensor contract"):
        module._load_capture(torch)
