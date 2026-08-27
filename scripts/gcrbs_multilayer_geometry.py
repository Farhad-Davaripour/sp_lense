from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gradient_specificity_v3_development as base
import global_counterfactual_robust_boundary_development as offline
from sp_lense.gcrbs_capture_adapter import adapt_v3_captures_to_gcrbs
from sp_lense.global_counterfactual_robust_boundary import (
    GCRBSError,
    solve_global_counterfactual_robust_boundary,
)

LOCK_SCHEMA = "sp_lense.gcrbs_multilayer_geometry_lock.v1"
LEDGER_SCHEMA = "sp_lense.gcrbs_multilayer_compute_ledger.v1"
CHUNK_SCHEMA = "sp_lense.gcrbs_multilayer_capture_chunk.v1"
CAPTURE_SCHEMA = "sp_lense.gcrbs_multilayer_capture_manifest.v1"
GEOMETRY_SCHEMA = "sp_lense.gcrbs_24_layer_geometry.v1"
LAYERS = tuple(range(24))
FISHER_TOP_COUNT = 8
COMPETITOR_COUNT = 8
EXPECTED_FORM_COUNT = 64
MAXIMUM_FORWARD_EVALUATIONS = 64
MAXIMUM_BACKWARD_EVALUATIONS = 64

LOCK_PATH = ROOT / "configs" / "gcrbs_multilayer_geometry_lock.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "global_counterfactual_robust_boundary" / "qwen35_08b"
CHUNK_ROOT = ARTIFACT_ROOT / "multilayer_capture_chunks"
LEDGER_PATH = ARTIFACT_ROOT / "multilayer_capture_compute_ledger.json"
CAPTURE_MANIFEST_PATH = ARTIFACT_ROOT / "multilayer_capture_manifest.json"
RESULT_PATH = (
    ROOT
    / "results"
    / "global_counterfactual_robust_boundary"
    / "qwen35_08b"
    / "all_layer_geometry.json"
)
SCRIPT_PATH = Path(__file__).resolve()


def _forms() -> list[dict[str, Any]]:
    forms = [
        *base.render_sp_forms("A", capture_scope=True),
        *base.render_unrelated_forms("nuisance_fit"),
    ]
    forms.sort(key=lambda form: str(form["form_id"]))
    if len(forms) != EXPECTED_FORM_COUNT:
        raise RuntimeError(
            f"multilayer geometry requires {EXPECTED_FORM_COUNT} forms, got {len(forms)}"
        )
    identifiers = [str(form["form_id"]) for form in forms]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("multilayer geometry form IDs are not unique")
    return forms


def _form_manifest(forms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in form.items() if key != "prompt"} for form in forms]


def _bound_paths() -> dict[str, Path]:
    return {
        "runner": SCRIPT_PATH,
        "v3_runner": base.SCRIPT_PATH,
        "v3_development_manifest": base.DEVELOPMENT_PATH,
        "backend": base.BACKEND_PATH,
        "runtime": base.RUNTIME_PATH,
        "intervention": base.INTERVENTION_PATH,
        "offline_lock": offline.LOCK_PATH,
        "offline_screen": offline.SCREEN_PATH,
    }


def _lock_payload() -> dict[str, Any]:
    offline_lock = offline._validate_lock()
    screen = offline._load_json(offline.SCREEN_PATH)
    if screen.get("lock_sha256") != offline_lock["lock_sha256"]:
        raise RuntimeError("layer-10 screen is not bound to the current offline lock")
    forms = _forms()
    value: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "status": "locked_before_multilayer_model_capture",
        "development_only": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "parent_lock_sha256": offline_lock["lock_sha256"],
        "parent_screen_sha256": screen["screen_sha256"],
        "source_and_parent_sha256": {
            key: offline.file_sha256(path) for key, path in _bound_paths().items()
        },
        "capture": {
            "layers": list(LAYERS),
            "hook_site": "blocks.{zero_based_layer}.hook_out",
            "position": "final_prompt_token",
            "gradient_coordinate": "residual_scaled_final_prompt",
            "fisher_top_count": FISHER_TOP_COUNT,
            "competitor_count": COMPETITOR_COUNT,
            "form_count": len(forms),
            "form_manifest_sha256": offline.canonical_sha256(_form_manifest(forms)),
            "maximum_forward_evaluations": MAXIMUM_FORWARD_EVALUATIONS,
            "maximum_backward_evaluations": MAXIMUM_BACKWARD_EVALUATIONS,
            "batched_vjp_all_layers_in_one_backward": True,
            "layer10_exact_recapture_equivalence_required": True,
        },
        "solver": {
            "required_margin": offline.REQUIRED_MARGIN,
            "residual_relative_l2_cap": offline.RESIDUAL_RELATIVE_L2_CAP,
            "aggregate_fisher_budget_each": offline.AGGREGATE_FISHER_BUDGET,
            "per_prompt_fisher_cap_each": offline.PER_PROMPT_FISHER_CAP,
            "all_24_layers_reported": True,
            "finite_model_scoring_in_this_phase": False,
        },
        "evaluation_policy": {
            "external_model_judges": 0,
            "external_api_calls": 0,
            "generated_tokens": 0,
            "sealed_data_viewed": False,
        },
    }
    value["lock_sha256"] = offline.canonical_sha256(value)
    return value


def _validate_lock() -> dict[str, Any]:
    lock = offline._load_json(LOCK_PATH)
    stored = lock.get("lock_sha256")
    unhashed = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("schema_version") != LOCK_SCHEMA or stored != offline.canonical_sha256(unhashed):
        raise RuntimeError("multilayer lock failed its internal hash")
    hashes = lock.get("source_and_parent_sha256")
    if not isinstance(hashes, Mapping):
        raise RuntimeError("multilayer lock has no source hashes")
    for key, path in _bound_paths().items():
        if hashes.get(key) != offline.file_sha256(path):
            raise RuntimeError(f"multilayer bound file changed: {key}")
    forms = _forms()
    if lock["capture"].get("form_manifest_sha256") != offline.canonical_sha256(
        _form_manifest(forms)
    ):
        raise RuntimeError("multilayer form manifest changed after lock")
    return lock


def run_lock() -> dict[str, Any]:
    proposed = _lock_payload()
    if LOCK_PATH.is_file():
        current = _validate_lock()
        excluded = {"created_utc", "lock_sha256"}
        proposed_stable = {key: value for key, value in proposed.items() if key not in excluded}
        current_stable = {key: value for key, value in current.items() if key not in excluded}
        if proposed_stable != current_stable:
            raise RuntimeError("existing multilayer lock differs from the proposed lock")
        return current
    offline._atomic_json(LOCK_PATH, proposed)
    return proposed


class ComputeLedger:
    def __init__(self, lock_sha256: str) -> None:
        self.lock_sha256 = lock_sha256
        if LEDGER_PATH.is_file():
            payload = offline._load_json(LEDGER_PATH)
            stored = payload.get("ledger_sha256")
            unhashed = {key: value for key, value in payload.items() if key != "ledger_sha256"}
            if payload.get("schema_version") != LEDGER_SCHEMA or stored != offline.canonical_sha256(
                unhashed
            ):
                raise RuntimeError("multilayer compute ledger failed its internal hash")
            if payload.get("lock_sha256") != lock_sha256:
                raise RuntimeError("multilayer compute ledger belongs to another lock")
            self.events = list(payload.get("events", []))
        else:
            self.events: list[dict[str, Any]] = []
            self._persist()
        previous = None
        work_ids: set[str] = set()
        for index, event in enumerate(self.events):
            stored = event.get("event_sha256")
            unhashed = {key: value for key, value in event.items() if key != "event_sha256"}
            if (
                event.get("sequence") != index
                or event.get("prior_event_sha256") != previous
                or stored != offline.canonical_sha256(unhashed)
                or event.get("work_id") in work_ids
            ):
                raise RuntimeError("multilayer compute ledger event chain is invalid")
            work_ids.add(str(event["work_id"]))
            previous = stored
        if self.count("forward") > MAXIMUM_FORWARD_EVALUATIONS or self.count(
            "backward"
        ) > MAXIMUM_BACKWARD_EVALUATIONS:
            raise RuntimeError("multilayer compute ledger exceeds its locked ceiling")

    def count(self, operation: str) -> int:
        return sum(event.get("operation") == operation for event in self.events)

    def has(self, work_id: str) -> bool:
        return any(event.get("work_id") == work_id for event in self.events)

    def reserve(self, work_id: str, operation: str) -> None:
        if operation not in {"forward", "backward"}:
            raise ValueError("ledger operation must be forward or backward")
        if self.has(work_id):
            raise RuntimeError(f"duplicate multilayer ledger work ID: {work_id}")
        ceiling = (
            MAXIMUM_FORWARD_EVALUATIONS
            if operation == "forward"
            else MAXIMUM_BACKWARD_EVALUATIONS
        )
        if self.count(operation) >= ceiling:
            raise RuntimeError(f"multilayer {operation} ceiling exhausted")
        event = {
            "sequence": len(self.events),
            "lock_sha256": self.lock_sha256,
            "work_id": work_id,
            "operation": operation,
            "prior_event_sha256": (
                self.events[-1]["event_sha256"] if self.events else None
            ),
        }
        event["event_sha256"] = offline.canonical_sha256(event)
        self.events.append(event)
        self._persist()

    def _persist(self) -> None:
        payload = {
            "schema_version": LEDGER_SCHEMA,
            "lock_sha256": self.lock_sha256,
            "maximum_forward_evaluations": MAXIMUM_FORWARD_EVALUATIONS,
            "maximum_backward_evaluations": MAXIMUM_BACKWARD_EVALUATIONS,
            "forward_evaluations": self.count("forward"),
            "backward_evaluations": self.count("backward"),
            "events": self.events,
        }
        payload["ledger_sha256"] = offline.canonical_sha256(payload)
        offline._atomic_json(LEDGER_PATH, payload, immutable=False)


def _capture_all_layers(backend: Any, form: Mapping[str, Any]) -> list[dict[str, Any]]:
    if {str(form["positive_label"]), str(form["negative_label"])} != {"A", "B"}:
        raise ValueError("GCRBS choices must use exactly A and B")
    torch = backend.torch
    prompt = str(form["prompt"])
    tokens = backend.encode(prompt)
    boundary = base.resolve_choice_boundary(backend, prompt)
    prompt_length = int(tokens.shape[-1])
    if boundary.prompt_length != prompt_length:
        raise RuntimeError("choice-boundary evidence has the wrong prompt length")
    positive_id = boundary.token_id(str(form["positive_label"]))
    negative_id = boundary.token_id(str(form["negative_label"]))
    activations: dict[int, Any] = {}
    hook_calls = {layer: 0 for layer in LAYERS}

    def hook_for(layer: int) -> Any:
        def capture_hook(activation: Any, hook: Any) -> Any:
            del hook
            hook_calls[layer] += 1
            if hook_calls[layer] != 1:
                raise RuntimeError(f"multilayer hook {layer} fired more than once")
            if not activation.requires_grad:
                activation.requires_grad_(True)
            activations[layer] = activation
            return activation

        return capture_hook

    backend.model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    hooks = [(f"blocks.{layer}.hook_out", hook_for(layer)) for layer in LAYERS]
    with torch.enable_grad(), backend.model.hooks(fwd_hooks=hooks):
        logits = backend.model(tokens)[0, -1].double()
        if tuple(sorted(activations)) != LAYERS:
            raise RuntimeError("multilayer hooks did not capture all 24 residual streams")
        prompt_final_index = prompt_length - 1
        for layer, activation in activations.items():
            if (
                activation.ndim != 3
                or int(activation.shape[0]) != 1
                or int(activation.shape[1]) != prompt_length
                or prompt_final_index != int(activation.shape[1]) - 1
            ):
                raise RuntimeError(f"layer {layer} residual activation shape is invalid")
        top9 = torch.topk(logits, k=COMPETITOR_COUNT + 1, largest=True, sorted=True)
        top9_ids = [int(value) for value in top9.indices.detach().tolist()]
        greedy_id = top9_ids[0]
        competitor_ids = top9_ids[1:]
        fisher_ids = list(top9_ids[:FISHER_TOP_COUNT])
        for token_id in (int(boundary.a_token_id), int(boundary.b_token_id)):
            if token_id not in fisher_ids:
                fisher_ids.append(token_id)
        log_z = torch.logsumexp(logits, dim=0)
        category_index = torch.tensor(fisher_ids, device=logits.device, dtype=torch.long)
        category_log_probs = logits[category_index] - log_z
        category_mask = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)
        category_mask[category_index] = False
        if not bool(category_mask.any().detach().item()):
            raise RuntimeError("multilayer Fisher aggregate tail is empty")
        tail_logsumexp = torch.logsumexp(logits[category_mask], dim=0)
        tail_log_probability = tail_logsumexp - log_z
        semantic_objective = logits[positive_id] - logits[negative_id]
        gap_objectives = torch.stack(
            [logits[greedy_id] - logits[token_id] for token_id in competitor_ids]
        )
        objectives = torch.cat(
            (
                semantic_objective.reshape(1),
                category_log_probs,
                tail_log_probability.reshape(1),
                gap_objectives,
            )
        )
        vjp_weights = torch.eye(
            int(objectives.numel()), device=objectives.device, dtype=objectives.dtype
        )
        batched_by_layer = torch.autograd.grad(
            objectives,
            tuple(activations[layer] for layer in LAYERS),
            grad_outputs=vjp_weights,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )

    if any(hook_calls[layer] != 1 for layer in LAYERS):
        raise RuntimeError("one or more multilayer hooks did not fire exactly once")
    elapsed = time.perf_counter() - started
    category_probabilities = category_log_probs.detach().exp().double().cpu().contiguous()
    tail_probability = float(tail_log_probability.detach().exp().item())
    probability_sum = float(category_probabilities.sum().item()) + tail_probability
    if not math.isclose(probability_sum, 1.0, rel_tol=2e-6, abs_tol=2e-6):
        raise RuntimeError(f"multilayer Fisher probabilities sum to {probability_sum}")
    semantic_log_odds = float(semantic_objective.detach().item())
    raw_a_minus_b = semantic_log_odds if str(form["positive_label"]) == "A" else -semantic_log_odds
    actual_label = (
        "A"
        if greedy_id == int(boundary.a_token_id)
        else "B"
        if greedy_id == int(boundary.b_token_id)
        else "OTHER"
    )
    actual_semantic = (
        "positive"
        if greedy_id == positive_id
        else "negative"
        if greedy_id == negative_id
        else "OTHER"
    )
    pair_semantic = "positive" if semantic_log_odds >= 0.0 else "negative"
    full_log_probs = torch.log_softmax(logits.detach().double(), dim=0).cpu()
    pair_mass = float(
        (full_log_probs[positive_id].exp() + full_log_probs[negative_id].exp()).item()
    )
    if semantic_log_odds >= 0.0:
        raw_probability = 1.0 / (1.0 + math.exp(-semantic_log_odds))
    else:
        exponential = math.exp(semantic_log_odds)
        raw_probability = exponential / (1.0 + exponential)
    probability_floor = 1e-15
    conditional_probability = min(1.0 - probability_floor, max(probability_floor, raw_probability))
    top9_union_required = list(top9_ids)
    for token_id in (int(boundary.a_token_id), int(boundary.b_token_id)):
        if token_id not in top9_union_required:
            top9_union_required.append(token_id)

    records: list[dict[str, Any]] = []
    category_count = len(fisher_ids)
    for layer, batched_gradients in zip(LAYERS, batched_by_layer, strict=True):
        activation = activations[layer]
        residual = activation[0, prompt_final_index].detach().float()
        residual_norm = residual.norm()
        if not bool(torch.isfinite(residual_norm).item()) or float(residual_norm.item()) <= 0.0:
            raise RuntimeError(f"layer {layer} residual norm is invalid")
        if batched_gradients.ndim != 4:
            raise RuntimeError(f"layer {layer} batched VJP shape is invalid")
        effective = (
            (batched_gradients[:, 0, prompt_final_index].detach().float() * residual_norm)
            .cpu()
            .contiguous()
        )
        if not bool(torch.isfinite(effective).all().item()):
            raise RuntimeError(f"layer {layer} residual-scaled gradients are non-finite")
        semantic_gradient = effective[0].contiguous()
        fisher_gradients = effective[1 : 1 + category_count].contiguous()
        tail_gradient = effective[1 + category_count].contiguous()
        gap_gradients = effective[2 + category_count :].contiguous()
        if tuple(gap_gradients.shape) != (COMPETITOR_COUNT, int(residual.numel())):
            raise RuntimeError(f"layer {layer} competitor gradient shape is invalid")
        records.append(
            {
                **{key: value for key, value in form.items() if key != "prompt"},
                "schema_version": base.CAPTURE_SCHEMA,
                "development_only": True,
                "gradient_coordinate": "residual_scaled_final_prompt",
                "objective_name": "semantic_positive_minus_negative_logit",
                "zero_based_layer": layer,
                "baseline_semantic_log_odds": semantic_log_odds,
                "baseline_raw_a_minus_b_log_odds": raw_a_minus_b,
                "baseline_actual_label": actual_label,
                "baseline_actual_semantic_choice": actual_semantic,
                "baseline_forced_pair_semantic_choice": pair_semantic,
                "baseline_answer_format_valid": actual_label != "OTHER",
                "baseline_correct": actual_semantic == "positive",
                "baseline_answer_pair_mass": pair_mass,
                "baseline_conditional_positive_probability": conditional_probability,
                "baseline_conditional_probability_numerically_clipped": (
                    conditional_probability != raw_probability
                ),
                "baseline_greedy_token_id": greedy_id,
                "choice_a_token_id": int(boundary.a_token_id),
                "choice_b_token_id": int(boundary.b_token_id),
                "choice_boundary_evidence_sha256": boundary.evidence_sha256,
                "prompt_token_ids_sha256": boundary.prompt_prefix_token_ids_sha256,
                "prompt_length": prompt_length,
                "prompt_final_index": prompt_final_index,
                "residual_norm": float(residual_norm.item()),
                "semantic_gradient": semantic_gradient,
                "semantic_gradient_sha256": base.v3.tensor_float32_sha256(semantic_gradient),
                "top9_token_ids": top9_ids,
                "top9_logit_values": [float(value) for value in top9.values.detach().tolist()],
                "top9_union_required_ab_token_ids": top9_union_required,
                "top9_union_required_ab_logit_values": [
                    float(logits[token_id].detach().item()) for token_id in top9_union_required
                ],
                "log_partition_logsumexp": float(log_z.detach().item()),
                "fisher_category_token_ids": fisher_ids,
                "fisher_category_probabilities": category_probabilities,
                "fisher_category_score_gradients": fisher_gradients,
                "fisher_category_score_gradients_sha256": base.v3.tensor_float32_sha256(
                    fisher_gradients
                ),
                "fisher_tail_probability": tail_probability,
                "fisher_tail_logsumexp": float(tail_logsumexp.detach().item()),
                "fisher_tail_log_probability": float(tail_log_probability.detach().item()),
                "fisher_tail_score_gradient": tail_gradient,
                "fisher_tail_score_gradient_sha256": base.v3.tensor_float32_sha256(
                    tail_gradient
                ),
                "greedy_competitor_token_ids": competitor_ids,
                "greedy_competitor_gap_gradients": gap_gradients,
                "greedy_competitor_gap_gradients_sha256": base.v3.tensor_float32_sha256(
                    gap_gradients
                ),
                "batched_vjp": True,
                "batched_vjp_objective_count": int(objectives.numel()),
                "hook_call_count": hook_calls[layer],
                "elapsed_seconds": elapsed,
                "all_layer_capture": True,
            }
        )
    backend.model.zero_grad(set_to_none=True)
    return records


def _chunk_paths(form_id: str) -> tuple[Path, Path]:
    stem = offline.canonical_sha256(form_id)[:24]
    return CHUNK_ROOT / f"{stem}.pt", CHUNK_ROOT / f"{stem}.json"


def _load_chunk(torch: Any, form_id: str, lock_sha256: str) -> dict[str, Any] | None:
    chunk_path, manifest_path = _chunk_paths(form_id)
    if chunk_path.exists() != manifest_path.exists():
        raise RuntimeError(f"partial multilayer chunk pair for {form_id}")
    if not chunk_path.exists():
        return None
    manifest = offline._load_json(manifest_path)
    if (
        manifest.get("schema_version") != CHUNK_SCHEMA
        or manifest.get("lock_sha256") != lock_sha256
        or manifest.get("form_id") != form_id
        or manifest.get("chunk_file_sha256") != offline.file_sha256(chunk_path)
    ):
        raise RuntimeError(f"multilayer chunk manifest failed for {form_id}")
    payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if (
        not isinstance(records, list)
        or len(records) != len(LAYERS)
        or tuple(record.get("zero_based_layer") for record in records) != LAYERS
    ):
        raise RuntimeError(f"multilayer chunk payload failed for {form_id}")
    return dict(payload)


def _save_chunk(torch: Any, form_id: str, lock_sha256: str, records: list[dict[str, Any]]) -> None:
    chunk_path, manifest_path = _chunk_paths(form_id)
    payload = {
        "schema_version": CHUNK_SCHEMA,
        "lock_sha256": lock_sha256,
        "form_id": form_id,
        "records": records,
    }
    base.atomic_torch_save(torch, chunk_path, payload)
    manifest = {
        "schema_version": CHUNK_SCHEMA,
        "lock_sha256": lock_sha256,
        "form_id": form_id,
        "record_count": len(records),
        "layers": list(LAYERS),
        "chunk_path": offline._relative(chunk_path),
        "chunk_file_sha256": offline.file_sha256(chunk_path),
        "record_tensor_hashes_sha256": offline.canonical_sha256(
            [
                {
                    "layer": record["zero_based_layer"],
                    "semantic": record["semantic_gradient_sha256"],
                    "fisher": record["fisher_category_score_gradients_sha256"],
                    "tail": record["fisher_tail_score_gradient_sha256"],
                    "gaps": record["greedy_competitor_gap_gradients_sha256"],
                }
                for record in records
            ]
        ),
    }
    manifest["manifest_sha256"] = offline.canonical_sha256(manifest)
    offline._atomic_json(manifest_path, manifest)


def _validate_layer10_equivalence(torch: Any, forms: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]) -> str:
    sp_frozen = offline._load_capture(torch, offline.SP_CAPTURE_PATH, offline.SP_MANIFEST_PATH)
    nuisance_frozen = offline._load_capture(
        torch, offline.NUISANCE_CAPTURE_PATH, offline.NUISANCE_MANIFEST_PATH
    )
    frozen = {
        str(record["form_id"]): record
        for record in (*sp_frozen["records"], *nuisance_frozen["records"])
    }
    compared: list[dict[str, Any]] = []
    fields = (
        "semantic_gradient_sha256",
        "fisher_category_score_gradients_sha256",
        "fisher_tail_score_gradient_sha256",
        "greedy_competitor_gap_gradients_sha256",
        "baseline_semantic_log_odds",
        "baseline_greedy_token_id",
        "top9_token_ids",
        "top9_logit_values",
    )
    for form in forms:
        form_id = str(form["form_id"])
        payload = _load_chunk(torch, form_id, str(lock["lock_sha256"]))
        assert payload is not None
        observed = payload["records"][offline.ZERO_BASED_LAYER]
        expected = frozen.get(form_id)
        if expected is None:
            raise RuntimeError(f"layer-10 frozen capture lacks {form_id}")
        for field in fields:
            if observed.get(field) != expected.get(field):
                raise RuntimeError(f"layer-10 recapture differs for {form_id}: {field}")
        compared.append({"form_id": form_id, **{field: observed[field] for field in fields[:4]}})
    return offline.canonical_sha256(compared)


def run_capture() -> dict[str, Any]:
    lock = _validate_lock()
    forms = _forms()
    ledger = ComputeLedger(str(lock["lock_sha256"]))
    import torch

    missing = [
        form
        for form in forms
        if _load_chunk(torch, str(form["form_id"]), str(lock["lock_sha256"])) is None
    ]
    backend = base.load_backend() if missing else None
    for form in missing:
        form_id = str(form["form_id"])
        forward_id = f"{form_id}:all_layers:forward"
        backward_id = f"{form_id}:all_layers:batched_backward"
        if ledger.has(forward_id) or ledger.has(backward_id):
            raise RuntimeError(f"orphaned charged operation prevents replay for {form_id}")
        ledger.reserve(forward_id, "forward")
        ledger.reserve(backward_id, "backward")
        assert backend is not None
        records = _capture_all_layers(backend, form)
        _save_chunk(torch, form_id, str(lock["lock_sha256"]), records)
        print(
            json.dumps(
                {
                    "captured": form_id,
                    "completed": ledger.count("forward"),
                    "total": EXPECTED_FORM_COUNT,
                    "elapsed_seconds": records[0]["elapsed_seconds"],
                },
                allow_nan=False,
            ),
            flush=True,
        )
    if ledger.count("forward") != EXPECTED_FORM_COUNT or ledger.count(
        "backward"
    ) != EXPECTED_FORM_COUNT:
        raise RuntimeError("multilayer capture ledger is incomplete")
    equivalence_sha256 = _validate_layer10_equivalence(torch, forms, lock)
    chunk_manifests = [
        offline._load_json(_chunk_paths(str(form["form_id"]))[1]) for form in forms
    ]
    record = {
        "schema_version": CAPTURE_SCHEMA,
        "status": "complete",
        "development_only": True,
        "lock_sha256": lock["lock_sha256"],
        "form_count": len(forms),
        "layers": list(LAYERS),
        "forward_evaluations": ledger.count("forward"),
        "backward_evaluations": ledger.count("backward"),
        "generated_tokens": 0,
        "layer10_exact_recapture_equivalence": True,
        "layer10_equivalence_sha256": equivalence_sha256,
        "chunk_manifest_sha256s": [manifest["manifest_sha256"] for manifest in chunk_manifests],
        "ledger_file_sha256": offline.file_sha256(LEDGER_PATH),
    }
    record["capture_manifest_sha256"] = offline.canonical_sha256(record)
    offline._atomic_json(CAPTURE_MANIFEST_PATH, record)
    return record


def _layer_records(torch: Any, forms: Sequence[Mapping[str, Any]], lock_sha256: str) -> dict[int, list[dict[str, Any]]]:
    output = {layer: [] for layer in LAYERS}
    for form in forms:
        payload = _load_chunk(torch, str(form["form_id"]), lock_sha256)
        if payload is None:
            raise RuntimeError("multilayer capture is incomplete")
        for record in payload["records"]:
            output[int(record["zero_based_layer"])].append(record)
    return output


def run_solve() -> dict[str, Any]:
    lock = _validate_lock()
    capture = offline._load_json(CAPTURE_MANIFEST_PATH)
    if capture.get("status") != "complete" or capture.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("complete lock-bound multilayer capture is required")
    import torch

    forms = _forms()
    records_by_layer = _layer_records(torch, forms, str(lock["lock_sha256"]))
    results: list[dict[str, Any]] = []
    for layer in LAYERS:
        records = records_by_layer[layer]
        sp_records = [record for record in records if record.get("family") == "self_preservation"]
        nuisance_records = [record for record in records if record.get("unrelated_role") == "nuisance_fit"]
        constraints = adapt_v3_captures_to_gcrbs(
            torch,
            sp_records=sp_records,
            nuisance_records=nuisance_records,
            required_margin=offline.REQUIRED_MARGIN,
        )
        labels, budgets = offline._solver_budget_schedule(constraints)
        try:
            solution = solve_global_counterfactual_robust_boundary(
                **constraints.solver_kwargs(group_metric_budgets=budgets),
                l2_cap=offline.RESIDUAL_RELATIVE_L2_CAP,
            )
        except GCRBSError as error:
            result = {
                "zero_based_layer": layer,
                "status": "solver_failed_closed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "eligible_for_finite_discovery_oracle": False,
                "adapter_provenance_sha256": constraints.provenance["provenance_sha256"],
            }
        else:
            result = {
                "zero_based_layer": layer,
                "status": "certified_affine_candidate",
                "gamma": solution.gamma,
                "eligible_for_finite_discovery_oracle": solution.gamma >= offline.REQUIRED_MARGIN,
                "direction": solution.direction.tolist(),
                "direction_float64_sha256": offline.float64_array_sha256(solution.direction),
                "adapter_provenance_sha256": constraints.provenance["provenance_sha256"],
                "group_metric_labels": list(labels),
                "group_metric_budgets": list(budgets),
                "solver_diagnostics": solution.diagnostics,
            }
        results.append(result)
        print(
            json.dumps(
                {
                    "solved_layer": layer,
                    "status": result["status"],
                    "gamma": result.get("gamma"),
                    "eligible": result["eligible_for_finite_discovery_oracle"],
                },
                allow_nan=False,
            ),
            flush=True,
        )
    frozen_screen = offline._load_json(offline.SCREEN_PATH)
    layer10 = results[offline.ZERO_BASED_LAYER]
    if (
        layer10.get("status") != "certified_affine_candidate"
        or layer10["solver_diagnostics"]["input_sha256"]
        != frozen_screen["solver_diagnostics"]["input_sha256"]
        or layer10.get("gamma") != frozen_screen.get("gamma")
    ):
        raise RuntimeError("multilayer layer-10 solve does not exactly reproduce frozen screen")
    eligible = [
        int(result["zero_based_layer"])
        for result in results
        if result["eligible_for_finite_discovery_oracle"]
    ]
    record = {
        "schema_version": GEOMETRY_SCHEMA,
        "status": "complete",
        "development_only": True,
        "lock_sha256": lock["lock_sha256"],
        "capture_manifest_sha256": capture["capture_manifest_sha256"],
        "required_margin": offline.REQUIRED_MARGIN,
        "eligible_layer_indices": eligible,
        "eligible_layer_count": len(eligible),
        "layer_results": results,
        "forward_evaluations": capture["forward_evaluations"],
        "backward_evaluations": capture["backward_evaluations"],
        "generated_tokens": 0,
        "finite_model_scoring_performed": False,
        "external_model_judges": 0,
        "external_api_calls": 0,
    }
    record["geometry_sha256"] = offline.canonical_sha256(record)
    offline._atomic_json(RESULT_PATH, record)
    return record


def _summary(command: str, value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "lock_sha256",
        "capture_manifest_sha256",
        "geometry_sha256",
        "eligible_layer_indices",
        "eligible_layer_count",
        "forward_evaluations",
        "backward_evaluations",
        "generated_tokens",
    )
    return {"command": command, **{key: value[key] for key in keys if key in value}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locked 24-layer GCRBS geometry phase")
    parser.add_argument("command", choices=("lock", "capture", "solve"))
    arguments = parser.parse_args(argv)
    if arguments.command == "lock":
        value = run_lock()
    elif arguments.command == "capture":
        value = run_capture()
    else:
        value = run_solve()
    print(json.dumps(_summary(arguments.command, value), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
