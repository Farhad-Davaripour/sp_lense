from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sp_lense.comparison_intervention import InterventionSpec

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import gradient_specificity_v3_development as base

LOCK_PATH = ROOT / "configs" / "gradient_specificity_v3_absolute_dose_probe_lock.json"
RESULT_ROOT = (
    ROOT
    / "results"
    / "gradient_specificity_v3_development"
    / "absolute_dose_probe_v1"
    / "qwen35_08b"
    / "stage_a"
)
ROW_SCHEMA = "sp_lense.gradient_specificity_v3_absolute_dose_row.v1"
ROWS_MANIFEST_SCHEMA = "sp_lense.gradient_specificity_v3_absolute_dose_rows_manifest.v1"
SUMMARY_SCHEMA = "sp_lense.gradient_specificity_v3_absolute_dose_summary.v1"
LOCK_SCHEMA = "sp_lense.gradient_specificity_v3_absolute_dose_probe_lock.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_lock() -> dict[str, Any]:
    lock = _load_json(LOCK_PATH)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("absolute-dose lock has the wrong schema")
    if lock.get("status") != "locked_before_absolute_dose_probe_execution":
        raise ValueError("absolute-dose probe was not locked before execution")
    if "cannot validate" not in str(lock.get("claim_boundary", "")).lower():
        raise ValueError("absolute-dose lock lacks its development-only claim boundary")
    doses = lock.get("absolute_residual_relative_doses")
    if doses != [0.02, 0.05, 0.1, 0.15]:
        raise ValueError("absolute-dose grid differs from the locked grid")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in doses):
        raise ValueError("absolute doses must be finite and positive")
    if lock.get("signs") != ["plus", "minus"]:
        raise ValueError("absolute-dose signs differ from the lock")
    if lock.get("model") != base.EXPECTED_MODEL:
        raise ValueError("absolute-dose model settings differ from v3")
    return lock


def _bank_paths(lock: Mapping[str, Any]) -> tuple[Path, Path]:
    bank_path = ROOT / str(lock["source"]["direction_bank"])
    manifest_path = bank_path.with_name("direction_bank_manifest.json")
    if not bank_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("frozen Stage-A direction bank is unavailable")
    return bank_path, manifest_path


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    lock = _load_lock()
    development = base.load_development_manifest()
    bank = base._load_complete_bank("A")
    entries = base._constructed_entries(bank)
    expected_count = int(lock["source"]["direction_count"])
    if len(entries) != expected_count:
        raise RuntimeError("frozen Stage-A constructed-direction count changed")
    if len({str(entry["direction_sha256"]) for entry in entries}) != len(entries):
        raise RuntimeError("frozen Stage-A direction hashes are not unique")
    return lock, development, entries


def _forms(family: str) -> list[dict[str, Any]]:
    if family == "sp":
        forms = base.render_sp_forms("A")
    elif family == "controls":
        forms = base.render_unrelated_forms("audit_control")
    else:
        raise ValueError("family must be sp or controls")
    return [dict(form) for form in forms]


def _jobs(
    family: str,
    entries: Sequence[Mapping[str, Any]],
    forms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if family == "sp":
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for form in forms:
            grouped[(str(form["case_id"]), int(form["assignment"]))].append(form)
        for entry in entries:
            key = (str(entry["case_id"]), int(entry["assignment"]))
            for form in grouped[key]:
                jobs.append(
                    {
                        "unit_id": (
                            f"{entry['direction_key']}::{form['target']}::"
                            f"preserve_{'A' if form['preserve_first'] else 'B'}"
                        ),
                        "entry": entry,
                        "form": form,
                    }
                )
    else:
        for entry in entries:
            for form in forms:
                jobs.append(
                    {
                        "unit_id": f"control::{entry['direction_key']}::{form['form_id']}",
                        "entry": entry,
                        "form": form,
                    }
                )
    if len({str(job["unit_id"]) for job in jobs}) != len(jobs):
        raise RuntimeError("absolute-dose jobs have duplicate unit IDs")
    return jobs


def _identity(
    family: str,
    *,
    lock: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    forms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bank_path, bank_manifest_path = _bank_paths(lock)
    payload = {
        "schema_version": "sp_lense.gradient_specificity_v3_absolute_dose_identity.v1",
        "development_only": True,
        "family": family,
        "lock_sha256": _file_sha256(LOCK_PATH),
        "script_sha256": _file_sha256(SCRIPT_PATH),
        "base_runner_sha256": _file_sha256(base.SCRIPT_PATH),
        "direction_bank_sha256": _file_sha256(bank_path),
        "direction_bank_manifest_sha256": _file_sha256(bank_manifest_path),
        "direction_hashes": sorted(str(entry["direction_sha256"]) for entry in entries),
        "form_manifest_sha256": base.canonical_sha256(base._form_manifest(forms)),
        "absolute_residual_relative_doses": list(lock["absolute_residual_relative_doses"]),
        "model": dict(lock["model"]),
    }
    payload["identity_sha256"] = base.canonical_sha256(payload)
    return payload


def run_preflight() -> dict[str, Any]:
    lock, _, entries = _load_inputs()
    output: dict[str, Any] = {
        "schema_version": "sp_lense.gradient_specificity_v3_absolute_dose_preflight.v1",
        "development_only": True,
        "status": "ready_for_development_only_execution",
        "lock_sha256": _file_sha256(LOCK_PATH),
        "script_sha256": _file_sha256(SCRIPT_PATH),
        "direction_count": len(entries),
        "absolute_residual_relative_doses": lock["absolute_residual_relative_doses"],
        "families": {},
    }
    for family in ("sp", "controls"):
        forms = _forms(family)
        jobs = _jobs(family, entries, forms)
        identity = _identity(family, lock=lock, entries=entries, forms=forms)
        output["families"][family] = {
            "form_count": len(forms),
            "job_count": len(jobs),
            "expected_row_count": len(jobs)
            * (1 + 2 * len(lock["absolute_residual_relative_doses"])),
            "identity_sha256": identity["identity_sha256"],
        }
    return output


def _score_unit(
    backend: Any,
    *,
    job: Mapping[str, Any],
    doses: Sequence[float],
    identity: Mapping[str, Any],
    baseline_cache: dict[str, tuple[dict[str, Any], Any]],
) -> list[dict[str, Any]]:
    form = job["form"]
    entry = job["entry"]
    prompt = str(form["prompt"])
    direction = entry["direction"].to(backend.device)
    cache_key = f"{form['prompt_sha256']}::{form['positive_label']}::{form['negative_label']}"
    cached = baseline_cache.get(cache_key)
    if cached is None:
        baseline_score, baseline_logits = base._score_logits(
            backend, form, spec=None, baseline_logits=None
        )
        baseline_cache[cache_key] = (baseline_score, baseline_logits)
    else:
        baseline_score, baseline_logits = cached
    common = {
        **{key: value for key, value in form.items() if key != "prompt"},
        "schema_version": ROW_SCHEMA,
        "development_only": True,
        "study_identity_sha256": identity["identity_sha256"],
        "unit_id": str(job["unit_id"]),
        "direction_key": str(entry["direction_key"]),
        "direction_sha256": str(entry["direction_sha256"]),
        "source_native_residual_relative_norm": float(entry["native_residual_relative_norm"]),
        "model_id": base.EXPECTED_MODEL["id"],
        "model_revision": base.EXPECTED_MODEL["revision"],
        "layer": base.EXPECTED_MODEL["layer_zero_based"],
        "position": base.EXPECTED_MODEL["position"],
        "magnitude_mode": base.EXPECTED_MODEL["magnitude_mode"],
    }
    rows = [
        {
            **common,
            "condition": "baseline",
            "sign": 0,
            "absolute_residual_relative_dose": 0.0,
            "signed_strength": 0.0,
            **baseline_score,
        }
    ]
    prompt_length = int(backend.encode(prompt).shape[-1])
    for dose in doses:
        dose = float(dose)
        for condition, sign in (("plus", 1), ("minus", -1)):
            signed_strength = sign * dose
            spec = InterventionSpec(
                layer=int(base.EXPECTED_MODEL["layer_zero_based"]),
                direction=direction,
                strength=signed_strength,
                geometry="matched_final_prompt",
                prompt_length=prompt_length,
                magnitude_mode="residual_relative",
            )
            score, _ = base._score_logits(
                backend,
                form,
                spec=spec,
                baseline_logits=baseline_logits,
            )
            realized = float(score["realized_mean_relative_perturbation_norm"])
            if score["realized_perturbed_position_count"] != 1 or not math.isclose(
                realized,
                dose,
                rel_tol=2e-5,
                abs_tol=2e-7,
            ):
                raise RuntimeError("absolute-dose intervention realization failed")
            rows.append(
                {
                    **common,
                    "condition": condition,
                    "sign": sign,
                    "absolute_residual_relative_dose": dose,
                    "signed_strength": signed_strength,
                    **score,
                }
            )
    return rows


def _expected_cells(doses: Sequence[float]) -> set[tuple[str, float]]:
    return {("baseline", 0.0)} | {
        (condition, float(dose)) for dose in doses for condition in ("plus", "minus")
    }


def _validate_chunk(
    rows: Sequence[Mapping[str, Any]],
    *,
    job: Mapping[str, Any],
    identity: Mapping[str, Any],
    doses: Sequence[float],
) -> None:
    observed: set[tuple[str, float]] = set()
    for row in rows:
        if row.get("schema_version") != ROW_SCHEMA or row.get("development_only") is not True:
            raise ValueError("absolute-dose row has the wrong schema or scope")
        if row.get("study_identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("absolute-dose row has the wrong identity")
        if row.get("unit_id") != job["unit_id"]:
            raise RuntimeError("absolute-dose row has the wrong unit ID")
        if row.get("direction_sha256") != job["entry"]["direction_sha256"]:
            raise RuntimeError("absolute-dose row has the wrong direction hash")
        cell = (str(row["condition"]), float(row["absolute_residual_relative_dose"]))
        if cell in observed:
            raise ValueError("absolute-dose chunk has a duplicate cell")
        observed.add(cell)
    if observed != _expected_cells(doses):
        raise ValueError("absolute-dose chunk lacks exact dose/sign coverage")


def _paths(family: str) -> tuple[Path, Path, Path]:
    return (
        RESULT_ROOT / f"{family}_chunks",
        RESULT_ROOT / f"{family}_rows.jsonl",
        RESULT_ROOT / f"{family}_rows_manifest.json",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run_score(family: str, backend: Any | None = None) -> list[dict[str, Any]]:
    lock, _, entries = _load_inputs()
    forms = _forms(family)
    jobs = _jobs(family, entries, forms)
    identity = _identity(family, lock=lock, entries=entries, forms=forms)
    doses = list(map(float, lock["absolute_residual_relative_doses"]))
    chunk_root, rows_path, manifest_path = _paths(family)
    if rows_path.exists() or manifest_path.exists():
        if not rows_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("absolute-dose rows and manifest must exist together")
        rows = _read_jsonl(rows_path)
        manifest = _load_json(manifest_path)
        if (
            manifest.get("schema_version") != ROWS_MANIFEST_SCHEMA
            or manifest.get("identity") != identity
            or manifest.get("rows_file_sha256") != _file_sha256(rows_path)
            or int(manifest.get("row_count", -1)) != len(rows)
            or int(manifest.get("job_count", -1)) != len(jobs)
        ):
            raise RuntimeError("completed absolute-dose rows differ from their manifest")
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["unit_id"])].append(row)
        if set(grouped) != {str(job["unit_id"]) for job in jobs}:
            raise RuntimeError("completed absolute-dose unit coverage changed")
        by_id = {str(job["unit_id"]): job for job in jobs}
        for unit_id, chunk in grouped.items():
            _validate_chunk(chunk, job=by_id[unit_id], identity=identity, doses=doses)
        return rows

    chunk_root.mkdir(parents=True, exist_ok=True)
    resident = base.load_backend() if backend is None else backend
    baseline_cache: dict[str, tuple[dict[str, Any], Any]] = {}
    output: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        unit_id = str(job["unit_id"])
        chunk_path = chunk_root / f"{base.canonical_sha256(unit_id)[:24]}.jsonl"
        if chunk_path.is_file():
            rows = _read_jsonl(chunk_path)
            _validate_chunk(rows, job=job, identity=identity, doses=doses)
        else:
            print(f"absolute dose {family} {index}/{len(jobs)}: {unit_id}", flush=True)
            rows = _score_unit(
                resident,
                job=job,
                doses=doses,
                identity=identity,
                baseline_cache=baseline_cache,
            )
            _validate_chunk(rows, job=job, identity=identity, doses=doses)
            _atomic_jsonl(chunk_path, rows)
        output.extend(rows)
    output.sort(
        key=lambda row: (
            str(row["unit_id"]),
            float(row["absolute_residual_relative_dose"]),
            str(row["condition"]),
        )
    )
    _atomic_jsonl(rows_path, output)
    manifest = {
        "schema_version": ROWS_MANIFEST_SCHEMA,
        "development_only": True,
        "status": "complete",
        "identity": identity,
        "job_count": len(jobs),
        "row_count": len(output),
        "rows_path": _relative(rows_path),
        "rows_file_sha256": _file_sha256(rows_path),
    }
    base.atomic_json(manifest_path, manifest)
    return output


def _nearest_rank_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _kl(values: Sequence[float]) -> dict[str, Any]:
    numbers = list(map(float, values))
    return {
        "row_count": len(numbers),
        "mean": statistics.fmean(numbers) if numbers else 0.0,
        "empirical_p95": _nearest_rank_quantile(numbers, 0.95),
        "maximum": max(numbers) if numbers else 0.0,
    }


def _triplets(
    rows: Sequence[Mapping[str, Any]], dose: float
) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        row_dose = float(row["absolute_residual_relative_dose"])
        if condition != "baseline" and not math.isclose(row_dose, dose, abs_tol=1e-12):
            continue
        grouped[str(row["unit_id"])][condition] = row
    for unit_id, conditions in grouped.items():
        if set(conditions) != {"baseline", "plus", "minus"}:
            raise RuntimeError(f"absolute-dose unit {unit_id} lacks a triplet")
    return dict(grouped)


def _change_counts(triplets: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, int]:
    exact = semantic = new_other = losses = gains = 0
    for conditions in triplets.values():
        baseline = conditions["baseline"]
        for condition in ("plus", "minus"):
            changed = conditions[condition]
            exact += int(
                int(changed["actual_next_token_token_id"])
                != int(baseline["actual_next_token_token_id"])
            )
            semantic += int(
                changed["actual_next_token_semantic_choice"]
                != baseline["actual_next_token_semantic_choice"]
            )
            new_other += int(
                baseline["actual_next_token_semantic_choice"] != "OTHER"
                and changed["actual_next_token_semantic_choice"] == "OTHER"
            )
            losses += int(bool(baseline["correct"]) and not bool(changed["correct"]))
            gains += int(not bool(baseline["correct"]) and bool(changed["correct"]))
    return {
        "exact_greedy_token_changes": exact,
        "semantic_decision_changes": semantic,
        "new_other_outputs": new_other,
        "correct_to_incorrect_changes": losses,
        "incorrect_to_correct_changes": gains,
    }


def _self_report(triplets: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    intended_movement = 0
    order_count = 0
    actual_intended_changes = 0
    by_direction: dict[str, list[Mapping[str, Mapping[str, Any]]]] = defaultdict(list)
    for conditions in triplets.values():
        baseline = conditions["baseline"]
        if baseline.get("target") != "self":
            continue
        order_count += 1
        plus = conditions["plus"]
        minus = conditions["minus"]
        intended_movement += int(
            float(plus["semantic_positive_log_odds"])
            > float(baseline["semantic_positive_log_odds"])
        )
        intended_movement += int(
            float(minus["semantic_positive_log_odds"])
            < float(baseline["semantic_positive_log_odds"])
        )
        actual_intended_changes += int(
            baseline["actual_next_token_semantic_choice"] == "negative"
            and plus["actual_next_token_semantic_choice"] == "positive"
        )
        actual_intended_changes += int(
            baseline["actual_next_token_semantic_choice"] == "positive"
            and minus["actual_next_token_semantic_choice"] == "negative"
        )
        by_direction[str(baseline["direction_key"])].append(conditions)
    successes = 0
    for units in by_direction.values():
        if len(units) != 2 or {bool(unit["baseline"]["preserve_first"]) for unit in units} != {
            True,
            False,
        }:
            raise RuntimeError("self direction lacks both answer orders")
        if (
            all(unit["plus"]["actual_next_token_semantic_choice"] == "positive" for unit in units)
            and all(
                unit["minus"]["actual_next_token_semantic_choice"] == "negative" for unit in units
            )
            and all(
                unit["plus"]["actual_next_token_semantic_choice"]
                != unit["baseline"]["actual_next_token_semantic_choice"]
                or unit["minus"]["actual_next_token_semantic_choice"]
                != unit["baseline"]["actual_next_token_semantic_choice"]
                for unit in units
            )
        ):
            successes += 1
    return {
        "self_order_form_count": order_count,
        "signed_self_row_count": 2 * order_count,
        "intended_logit_movements": intended_movement,
        "actual_intended_A_B_changes": actual_intended_changes,
        "directions_meeting_both_signs_both_orders": successes,
    }


def summarize(
    *,
    lock: Mapping[str, Any],
    sp_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    limits = lock["protected_limits"]
    dose_reports = []
    passing_doses: list[float] = []
    for dose in map(float, lock["absolute_residual_relative_doses"]):
        sp = _triplets(sp_rows, dose)
        controls = _triplets(control_rows, dose)
        self_units = {
            unit_id: value
            for unit_id, value in sp.items()
            if value["baseline"].get("target") == "self"
        }
        other_units = {
            unit_id: value
            for unit_id, value in sp.items()
            if value["baseline"].get("target") == "other"
        }
        protected = {**other_units, **controls}
        protected_changes = _change_counts(protected)
        protected_changed_rows = [
            conditions[condition]
            for conditions in protected.values()
            for condition in ("plus", "minus")
        ]
        protected_kl = _kl(
            [row["full_vocabulary_kl_changed_to_baseline"] for row in protected_changed_rows]
        )
        passes = (
            protected_changes["exact_greedy_token_changes"]
            <= int(limits["maximum_exact_greedy_token_changes"])
            and protected_changes["new_other_outputs"] <= int(limits["maximum_new_other_outputs"])
            and protected_kl["mean"] <= float(limits["mean_full_vocabulary_kl_changed_to_baseline"])
            and protected_kl["empirical_p95"]
            <= float(limits["empirical_p95_full_vocabulary_kl_changed_to_baseline"])
            and protected_kl["maximum"]
            <= float(limits["maximum_full_vocabulary_kl_changed_to_baseline"])
        )
        if passes:
            passing_doses.append(dose)
        dose_reports.append(
            {
                "absolute_residual_relative_dose": dose,
                "protected_limits_pass": passes,
                "protected": {
                    "unit_count": len(protected),
                    "signed_row_count": len(protected_changed_rows),
                    "change_counts": protected_changes,
                    "full_vocabulary_kl_changed_to_baseline": protected_kl,
                },
                "matched_other": {
                    "change_counts": _change_counts(other_units),
                    "full_vocabulary_kl_changed_to_baseline": _kl(
                        [
                            conditions[condition]["full_vocabulary_kl_changed_to_baseline"]
                            for conditions in other_units.values()
                            for condition in ("plus", "minus")
                        ]
                    ),
                },
                "audit_controls": {
                    "change_counts": _change_counts(controls),
                    "full_vocabulary_kl_changed_to_baseline": _kl(
                        [
                            conditions[condition]["full_vocabulary_kl_changed_to_baseline"]
                            for conditions in controls.values()
                            for condition in ("plus", "minus")
                        ]
                    ),
                },
                "self": {
                    **_self_report(self_units),
                    "change_counts": _change_counts(self_units),
                    "full_vocabulary_kl_changed_to_baseline": _kl(
                        [
                            conditions[condition]["full_vocabulary_kl_changed_to_baseline"]
                            for conditions in self_units.values()
                            for condition in ("plus", "minus")
                        ]
                    ),
                },
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "development_only": True,
        "status": "complete",
        "lock_sha256": _file_sha256(LOCK_PATH),
        "trust_radius_selection_uses_self_outcomes": False,
        "selected_empirical_trust_radius": max(passing_doses) if passing_doses else None,
        "no_supported_positive_radius_on_grid": not passing_doses,
        "dose_reports": dose_reports,
    }
    summary["summary_sha256"] = base.canonical_sha256(summary)
    return summary


def _markdown(summary: Mapping[str, Any]) -> str:
    selected = summary["selected_empirical_trust_radius"]
    lines = [
        "# V3 absolute-dose probe result",
        "",
        "Status: development-only; previously opened prompts and frozen failed directions.",
        "",
        (
            f"The protected-only rule selected an empirical trust radius of `{selected}`."
            if selected is not None
            else "No tested positive dose satisfied every protected limit."
        ),
        "",
        (
            "| Absolute dose | Protected pass | Protected exact changes | Protected mean KL | "
            "Protected p95 KL | Protected max KL | Self intended A/B changes | "
            "Full direction successes |"
        ),
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in summary["dose_reports"]:
        protected = report["protected"]
        self_report = report["self"]
        kl = protected["full_vocabulary_kl_changed_to_baseline"]
        lines.append(
            f"| {report['absolute_residual_relative_dose']:.2f} | "
            f"{'yes' if report['protected_limits_pass'] else 'no'} | "
            f"{protected['change_counts']['exact_greedy_token_changes']} | "
            f"{kl['mean']:.6f} | {kl['empirical_p95']:.6f} | {kl['maximum']:.6f} | "
            f"{self_report['actual_intended_A_B_changes']} | "
            f"{self_report['directions_meeting_both_signs_both_orders']} |"
        )
    lines.extend(
        [
            "",
            (
                "The radius was selected without looking at self-target efficacy. This result "
                "does not revise the frozen Stage-A failure or support a confirmatory claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_report() -> dict[str, Any]:
    lock = _load_lock()
    _, sp_path, sp_manifest_path = _paths("sp")
    _, controls_path, controls_manifest_path = _paths("controls")
    for path in (sp_path, sp_manifest_path, controls_path, controls_manifest_path):
        if not path.is_file():
            raise RuntimeError("absolute-dose report requires both completed score families")
    if _load_json(sp_manifest_path).get("rows_file_sha256") != _file_sha256(sp_path):
        raise RuntimeError("absolute-dose SP rows changed after scoring")
    if _load_json(controls_manifest_path).get("rows_file_sha256") != _file_sha256(controls_path):
        raise RuntimeError("absolute-dose control rows changed after scoring")
    summary = summarize(
        lock=lock,
        sp_rows=_read_jsonl(sp_path),
        control_rows=_read_jsonl(controls_path),
    )
    base.atomic_json(RESULT_ROOT / "absolute_dose_summary.json", summary)
    report_path = RESULT_ROOT / "ABSOLUTE_DOSE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(_markdown(summary), encoding="utf-8", newline="\n")
    os.replace(temporary, report_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V3 frozen-direction absolute-dose safety probe")
    parser.add_argument(
        "command",
        choices=("preflight", "score-sp", "score-controls", "report", "all"),
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = run_preflight()
    elif arguments.command == "score-sp":
        result = run_score("sp")
    elif arguments.command == "score-controls":
        result = run_score("controls")
    elif arguments.command == "report":
        result = run_report()
    else:
        preflight = run_preflight()
        if preflight["status"] != "ready_for_development_only_execution":
            raise RuntimeError("absolute-dose preflight did not pass")
        backend = base.load_backend()
        run_score("sp", backend)
        run_score("controls", backend)
        result = run_report()
    if arguments.command in {"score-sp", "score-controls"}:
        result = {"status": "complete", "development_only": True, "row_count": len(result)}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
