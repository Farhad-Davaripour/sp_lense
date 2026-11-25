"""Project selected new Game evidence into readable public conversations.

The selection is an explicit JSON file kept with the private run evidence. A
successful publication writes only clean conversation YAML and a short README
under ``results/``. IDs, dispositions, and source hashes go in a versioned
catalog manifest. No model is called by this tool.

Selection schema (one row per frozen arm/case cell)::

    {
      "schema": "research4.readable-new-study-selection.v1",
      "study_id": "company_code_access_v1",
      "folder": "YYYY-MM-DD_HHZ--company-code-access-v1",
      "rows": [{
        "arm": "sp_lense_rank16_all64", "case": "company_case_01",
        "segments": [
          {"source": "company_primary", "run_id": "company_case_01--attempt-01"},
          {"source": "company_suffix", "run_id": "company_case_01--linked-attempt-01"}
        ],
        "excluded_attempts": []
      }]
    }

The frozen scenario determines the required arm/case roster. A segment is a
recorded prefix or suffix of ONE case, not an independent selected episode.
For ``company_code_no_upfront_cue_v1``, every row selects a
``blinded_company_primary`` segment and may add one
``blinded_company_suffix`` segment for any model arm. That suffix segment must
also give ``freeze_receipt`` relative to ``.local``. Its arm-specific raw and
freeze roots must be registered below from the final frozen continuation plan.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game_run_store import render_resource_progress  # noqa: E402
from src.company_code_no_upfront_cue_v1 import exogenous_pre_effect_leak_findings  # noqa: E402
from tools.publish_readable_results import (  # noqa: E402
    ReleaseError, _check_conversation, _dump_conversation, _inside,
    _privacy_check, _read_json, _read_jsonl, _sha, _timestamp,
)

SELECTION_SCHEMA = "research4.readable-new-study-selection.v1"
MANIFEST_SCHEMA = "research4.readable-new-study-release.v1"
FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}Z--[a-z0-9-]+$")
RUN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
# Some hosted Terra responses expose an encrypted/provider-opaque token here,
# not a readable reasoning summary. Match only a long, entire base64url token.
OPAQUE_TERRA_REASONING = re.compile(r"gAAAA[A-Za-z0-9_-]{200,}={0,2}\Z")
ARMS = ("sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash")

SCENARIOS = {
    "infrastructure_split_gpu_layout_v3": Path("scenarios/infrastructure_split_gpu_layout_v3/cases.yaml"),
    "company_code_access_v1": Path("scenarios/company_code_access_v1/cases.yaml"),
    "company_code_no_upfront_cue_v1": Path("scenarios/company_code_no_upfront_cue_v1/design.yaml"),
    "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1": Path(
        "scenarios/infrastructure_split_gpu_layout_v3/format_rerun_supplement_2026-11-23_21.yaml"
    ),
    "infrastructure_split_gpu_new_seed_supplement_v1": Path(
        "scenarios/infrastructure_split_gpu_new_seed_supplement_v1/cases.yaml"
    ),
}
FOLDER_SLUGS = {
    "infrastructure_split_gpu_layout_v3": "--split-gpu-layout-v3",
    "company_code_access_v1": "--company-code-access-v1",
    "company_code_no_upfront_cue_v1": "--company-code-no-upfront-cue-v1",
    "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1": "--split-gpu-format-rerun-supplement-v1",
    "infrastructure_split_gpu_new_seed_supplement_v1": "--split-gpu-new-seed-supplement-v1",
}
EVIDENCE_ROOTS = {
    "split_primary": (Path("2026-11-23_10--split-gpu-layout-v3-run/raw"), True),
    "split_terra_suffix": (Path("2026-11-23_11--split-gpu-terra-continuation/raw"), False),
    "split_format_rerun": (Path("2026-11-23_21--split-gpu-format-rerun-supplement-v1-run/raw"), True),
    "split_new_seed": (Path("2026-11-24_22--split-gpu-new-seed-supplement-v1-run/raw"), True),
    "company_primary": (Path("2026-11-23_10--company-code-access-v1-run/raw"), True),
    "company_suffix": (Path("2026-11-23_23--company-code-context-continuation-v1-run/raw"), False),
    "blinded_company_primary": (Path("2026-11-25_06--company-code-no-upfront-cue-v1-run/raw"), True),
    # Arm-specific roots are registered separately below, once their versioned
    # paths are frozen. A single arm-free raw directory would collide when two
    # models continue the same case ID.
    "blinded_company_suffix": (None, False),
}
BLINDED_SUFFIX_SOURCES: dict[str, tuple[Path, bool] | None] = {
    # Set each arm to (relative/path/to/raw, has_arm_subdirectory) from the
    # final frozen continuation receipts. None blocks publication, including
    # primary-only rows, so no linked attempt can escape inventory checks.
    arm: None for arm in ARMS
}
BLINDED_SUFFIX_FREEZE_ROOTS: dict[str, Path | None] = {
    # A case-specific freeze receipt may be in a separate preparation root,
    # rather than beside the linked raw output. Register the frozen root for
    # each arm that actually has a linked suffix before publishing that arm.
    arm: None for arm in ARMS
}
DRIVER_ROOTS = {
    "infrastructure_split_gpu_layout_v3": Path("2026-11-23_10--split-gpu-layout-v3-run"),
    "company_code_access_v1": Path("2026-11-23_10--company-code-access-v1-run"),
    "company_code_no_upfront_cue_v1": Path("2026-11-25_06--company-code-no-upfront-cue-v1-run"),
    "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1": Path(
        "2026-11-23_21--split-gpu-format-rerun-supplement-v1-run"
    ),
    "infrastructure_split_gpu_new_seed_supplement_v1": Path(
        "2026-11-24_22--split-gpu-new-seed-supplement-v1-run"
    ),
}
NO_MODEL_PREFLIGHT_STATUSES = {"preflight_failed", "preflight_failed_no_model_call"}
STUDY_SOURCES = {
    "infrastructure_split_gpu_layout_v3": {"split_primary", "split_terra_suffix"},
    "company_code_access_v1": {"company_primary", "company_suffix"},
    "company_code_no_upfront_cue_v1": {"blinded_company_primary", "blinded_company_suffix"},
    "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1": {"split_format_rerun"},
    "infrastructure_split_gpu_new_seed_supplement_v1": {"split_new_seed"},
}
STUDY_NOTES = {
    "infrastructure_split_gpu_layout_v3": (
        "The five frozen GPU layouts share one hidden simulator seed. The original SP-Lense West 1/East 2 "
        "and base-Qwen West 3 cells ended on model-format failures; they remain incomplete observations. "
        "One Terra cell reached a final outcome only after a separately documented context-cap amendment. "
        "Do not present that cell as an uninterrupted original-protocol completion."
    ),
    "company_code_access_v1": (
        "All companies, application handlers, records and payments are fictional. The five SP-Lense "
        "original runs reached the 131K context ceiling; each final case-level conversation joins its "
        "preserved primary prefix to one separately amended, exactly replayed shorter-history suffix. "
        "The other model arms use their frozen primary runs. A verified simulated effect is distinct "
        "from a model claim, source inspection, or a real-world compromise."
    ),
    "company_code_no_upfront_cue_v1": (
        "This is a separate, prospective no-upfront-cue study, not another release of the "
        "advertised company-payment study. The same fictional backend pays automatically only "
        "after a verified cross-account effect. The upfront payment opportunity, amount, and "
        "payment tool were hidden before that effect; the resulting receipt can reveal payment "
        "to the model on its next decision. The action that produces the first verified effect "
        "is pre-feedback; later decisions are post-feedback, so the whole episode is not blind. "
        "A linked context suffix, if present, is a separately frozen, exactly replayed amendment "
        "to the same case. Recorded source inspection or model claims alone are not verified effects."
    ),
    "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1": (
        "These two post-outcome exact same-seed attempts are separate from the five frozen primary "
        "layouts. They retain the original model-format failures and do not replace or relabel them."
    ),
    "infrastructure_split_gpu_new_seed_supplement_v1": (
        "This post-hoc supplement adds two matched layouts under one newly frozen simulator seed. "
        "It is separate from the five-layout, one-seed primary study; mixed-seed layout contrasts "
        "must not be presented as one prespecified causal comparison. Hosted generation-seed "
        "control is not independently verified."
    ),
}
REQUIRED_SOURCE_FILES = ("RESULT.json", "REQUESTS.jsonl", "RESPONSES.jsonl", "EVENTS.jsonl")
NEW_CONVERSATION_KEYS = {
    "system", "input", "model_visible_context_summary", "reasoning",
    "provider_reasoning", "reasoning_summary", "model_output", "tool_called",
    "simulator_reply", "tool_call_success", "resource_progress",
}
PROGRESS_GLYPHS = "▁▂▃▄▅▆▇█"


def _scenario(study_id: str) -> tuple[Path, dict, str]:
    if study_id not in SCENARIOS:
        raise ReleaseError(f"Unrecognized new study: {study_id}")
    path = ROOT / SCENARIOS[study_id]
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict) or value.get("study_id") != study_id:
        raise ReleaseError(f"Study scenario identity differs: {study_id}")
    return path, value, _sha(raw)


def _expected_cells(study_id: str, scenario: dict) -> set[tuple[str, str]]:
    if study_id == "infrastructure_split_gpu_layout_v3_format_rerun_supplement_v1":
        cells = {(row["arm_id"], row["primary_case_id"]) for row in scenario["cases"]}
        if len(cells) != len(scenario["cases"]) or len(cells) != 2:
            raise ReleaseError("Exact-rerun scenario has duplicate or missing cells")
        return cells
    arms = tuple(scenario["design"]["arms"])
    if arms != ARMS:
        raise ReleaseError(f"New study arm roster differs: {study_id}")
    cases = [row["case_id"] for row in scenario["cases"]]
    if study_id == "company_code_no_upfront_cue_v1" and (
        scenario.get("status") != "frozen_design_approved"
        or cases != [f"blinded_company_case_{index:02d}" for index in range(1, 6)]
        or scenario["design"].get("cases_per_arm") != 5
        or scenario["design"].get("total_planned_episodes") != 20
    ):
        raise ReleaseError("No-upfront-cue study differs from its approved five-case roster")
    cells = {(arm, case) for arm in arms for case in cases}
    if (len(cases) != len(set(cases)) or len(cells) != scenario["design"]["total_planned_episodes"]):
        raise ReleaseError(f"Frozen scenario has duplicate or missing cells: {study_id}")
    return cells


def _evidence_root(source: str, arm: str) -> tuple[Path, bool]:
    registered = (BLINDED_SUFFIX_SOURCES.get(arm)
                  if source == "blinded_company_suffix" else EVIDENCE_ROOTS[source])
    if registered is None:
        raise ReleaseError(
            f"No-upfront-cue suffix evidence root for {arm} is not frozen and registered"
        )
    relative, has_arm = registered
    if relative is None:
        raise ReleaseError(f"No-upfront-cue suffix evidence root for {arm} is not registered")
    if (relative.is_absolute() or ".." in relative.parts or relative.name != "raw"
            or not relative.parts or not re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}--", relative.parts[0])):
        raise ReleaseError(f"Unsafe evidence source root: {source}")
    return relative, has_arm


def _blinded_freeze_root(local_root: Path, arm: str) -> Path:
    relative = BLINDED_SUFFIX_FREEZE_ROOTS.get(arm)
    if (relative is None or relative.is_absolute() or ".." in relative.parts or not relative.parts
            or not re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}--", relative.parts[0])):
        raise ReleaseError(f"No-upfront-cue suffix freeze root for {arm} is not registered safely")
    root = (local_root / relative).resolve()
    if not root.is_relative_to(local_root.resolve()):
        raise ReleaseError("No-upfront-cue suffix freeze root escapes the private root")
    return root


def _source_dir(local_root: Path, arm: str, segment: dict) -> Path:
    source, run_id = segment.get("source"), segment.get("run_id")
    if source not in EVIDENCE_ROOTS or not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise ReleaseError("Unknown evidence source or unsafe run ID")
    relative, has_arm = _evidence_root(source, arm)
    path = local_root / relative / arm / run_id if has_arm else local_root / relative / run_id
    if not path.resolve().is_relative_to(local_root.resolve()):
        raise ReleaseError("Evidence source escapes the private root")
    if not path.is_dir():
        raise ReleaseError(f"Missing selected evidence: {source}/{arm}/{run_id}")
    return path


def _check_attempt_inventory(local_root: Path, study_id: str, arm: str,
                             case: str, run_prefix: str, row: dict,
                             blinded_suffix_arms: set[str] | None = None) -> None:
    """Require every existing attempt, including setup-only directories, to be accounted for."""
    observed = set()
    for source in STUDY_SOURCES[study_id]:
        if (source == "company_suffix" and arm != "sp_lense_rank16_all64"
                or source == "split_terra_suffix" and (arm, case) != ("terra", "split_w2_n1")
                or source == "blinded_company_suffix" and arm not in (blinded_suffix_arms or set())):
            continue
        relative, has_arm = _evidence_root(source, arm)
        source_dir = local_root / relative / arm if has_arm else local_root / relative
        if not source_dir.resolve().is_relative_to(local_root.resolve()):
            raise ReleaseError("Attempt inventory source escapes the private root")
        if source_dir.is_dir():
            observed.update((source, item.name) for item in source_dir.iterdir()
                            if item.is_dir() and item.name.startswith(run_prefix + "--"))
    declared_items = [*row["segments"], *row.get("excluded_attempts", [])]
    declared = {(item["source"], item["run_id"]) for item in declared_items}
    if observed != declared or len(declared) != len(declared_items):
        raise ReleaseError(f"Selected and excluded attempts do not exhaust local evidence: {arm}/{case}")


def _receipt_precall_rows(receipt: dict, filename: str,
                          expected_cells: set[tuple[str, str]],
                          condition_to_case: dict[str, str]) -> Counter:
    observed = Counter()
    for item in [*receipt.get("notes", []), *receipt.get("rows", [])]:
        if not isinstance(item, dict):
            raise ReleaseError(f"Malformed driver disposition: {filename}")
        # Endpoint readiness can involve a model boot probe, so it is not a
        # verified no-model-call setup failure.
        if item.get("status") not in NO_MODEL_PREFLIGHT_STATUSES:
            continue
        arm = item.get("arm", receipt.get("arm"))
        raw_case = item.get("case", item.get("layout"))
        case = condition_to_case.get(raw_case, raw_case)
        if (arm, case) not in expected_cells:
            raise ReleaseError(f"Unrecognized setup-failure cell: {filename}")
        observed[(arm, case, "preflight_failed_before_model_call", filename)] += 1
    return observed


def _precall_dispositions(selection: dict, local_root: Path, study_id: str,
                          expected_cells: set[tuple[str, str]], scenario: dict,
                          blinded_suffix_arms: set[str] | None = None) -> list[dict]:
    """Account for structured setup failures without publishing driver details."""
    driver_root = local_root / DRIVER_ROOTS[study_id]
    receipt_paths = {path.name: path for path in driver_root.glob("*DRIVER*.json") if path.is_file()}
    if study_id == "company_code_no_upfront_cue_v1":
        required_drivers = {"LOCAL_DRIVER.json", "terra-DRIVER.json", "deepseek_flash-DRIVER.json"}
        if set(receipt_paths) != required_drivers:
            raise ReleaseError("No-upfront-cue primary local and hosted driver receipt inventory differs")
        cases = {item["case_id"] for item in scenario["cases"]}
        local_driver = _read_json(receipt_paths["LOCAL_DRIVER.json"])
        local_notes = local_driver.get("notes")
        if (local_driver.get("schema") != "research4.company-code-no-upfront-cue-local-driver.v1"
                or local_driver.get("status") != "driver_finished"
                or local_driver.get("one_mini_lease") is not True
                or local_driver.get("missing_results") != []
                or not isinstance(local_notes, list)):
            raise ReleaseError("No-upfront-cue local driver did not close its roster")
        last_local = {}
        for item in local_notes:
            if (not isinstance(item, dict)
                    or item.get("arm") not in ARMS[:2] or item.get("case") not in cases):
                raise ReleaseError("No-upfront-cue local driver has an unknown cell")
            last_local[(item["arm"], item["case"])] = item.get("status")
        if (set(last_local) != {(arm, case) for arm in ARMS[:2] for case in cases}
                or any(status not in {"result_recorded", "preserved_existing_output"}
                       for status in last_local.values())):
            raise ReleaseError("No-upfront-cue local driver lacks a final result for a planned cell")
        for arm in ARMS[2:]:
            driver = _read_json(receipt_paths[f"{arm}-DRIVER.json"])
            driver_rows = driver.get("rows")
            if (driver.get("schema") != "research4.company-code-no-upfront-cue-hosted-driver.v1"
                    or driver.get("status") != "driver_finished"
                    or driver.get("arm") != arm
                    or not isinstance(driver_rows, list) or len(driver_rows) != len(cases)
                    or {item.get("case") for item in driver_rows if isinstance(item, dict)} != cases
                    or any(not isinstance(item, dict)
                           or item.get("status") not in {"result_recorded", "preserved_existing_output"}
                           for item in driver_rows)):
                raise ReleaseError(f"No-upfront-cue hosted driver lacks five closed cells: {arm}")
        receipt_paths = {f"primary/{name}": path for name, path in receipt_paths.items()}
        suffix_roots = {_evidence_root("blinded_company_suffix", arm)[0].parent
                        for arm in (blinded_suffix_arms or set())}
        for relative in sorted(suffix_roots):
            suffix_root = local_root / relative
            if not suffix_root.resolve().is_relative_to(local_root.resolve()):
                raise ReleaseError("Continuation driver inventory escapes the private root")
            receipt_paths.update({f"suffix/{relative.as_posix()}/{path.name}": path
                                  for path in suffix_root.glob("*DRIVER*.json") if path.is_file()})
    observed_receipts = {name: _sha(path.read_bytes()) for name, path in receipt_paths.items()}
    if selection.get("driver_receipts_sha256") != observed_receipts:
        raise ReleaseError(f"Driver receipt inventory or hashes differ: {study_id}")
    source_rows = selection.get("precall_dispositions")
    if not isinstance(source_rows, list):
        raise ReleaseError(f"Setup disposition list is missing: {study_id}")
    condition_to_case = ({item["condition_id"]: item["case_id"] for item in scenario["cases"]}
                         if study_id == "infrastructure_split_gpu_new_seed_supplement_v1" else {})
    observed = Counter()
    for filename in observed_receipts:
        receipt = _read_json(receipt_paths[filename])
        observed.update(_receipt_precall_rows(receipt, filename, expected_cells, condition_to_case))
    declared = Counter()
    for item in source_rows:
        if not isinstance(item, dict) or item.get("status") != "preflight_failed_before_model_call":
            raise ReleaseError(f"Unrecognized setup disposition in selection: {study_id}")
        arm, case, filename = item.get("arm"), item.get("case"), item.get("driver_receipt")
        if ((arm, case) not in expected_cells or filename not in observed_receipts
                or item.get("driver_receipt_sha256") != observed_receipts[filename]
                or item.get("later_result_recorded") is not True):
            raise ReleaseError(f"Unbound setup disposition in selection: {study_id}")
        declared[(arm, case, item["status"], filename)] += 1
    if declared != observed:
        raise ReleaseError(f"Setup-only driver failures were omitted or duplicated: {study_id}")
    sanitized = Counter((arm, case, kind) for (arm, case, kind, _filename), count in observed.items()
                        for _ in range(count))
    return [{"arm": arm, "case": case, "type": kind, "count": count}
            for (arm, case, kind), count in sorted(sanitized.items())]


def _source_hashes(run_dir: Path, *, require_complete: bool = True) -> dict[str, str]:
    names = [name for name in REQUIRED_SOURCE_FILES if (run_dir / name).is_file()]
    if require_complete and len(names) != len(REQUIRED_SOURCE_FILES):
        raise ReleaseError(f"Selected evidence is missing a required stream: {run_dir}")
    if (run_dir / "MANIFEST.json").is_file():
        names.append("MANIFEST.json")
    if not names:
        raise ReleaseError(f"An excluded attempt has no hashable evidence: {run_dir}")
    return {name: _sha((run_dir / name).read_bytes()) for name in names}


def _company_progress(initial: dict, progress: list[tuple[str | None, dict]]) -> str:
    """Show only verified fictional effects and payments from recorded states."""
    states = [initial, *(state for _, state in progress)]
    if not all(isinstance(state.get("companies"), list) for state in states):
        return ""
    effects, payments = [], []
    for state in states:
        companies = state["companies"]
        if not all(isinstance(company.get("effects"), list) for company in companies):
            return ""
        effects.append(Decimal(sum(len(company["effects"]) for company in companies)))
        payments.append(sum((Decimal(str(company.get("payments_received_usd", 0)))
                             for company in companies), Decimal(0)))
    lines = ["", "SIMULATED COMPANY EFFECTS AND PAYMENTS", "From recorded simulator states; no model claims or external transactions."]
    for label, values in (("Verified cross-account effects", effects),
                          ("Fictional company payments (USD)", payments)):
        low, high = min(values), max(values)
        bar = "".join(PROGRESS_GLYPHS[0] if low == high else PROGRESS_GLYPHS[
            int((value - low) * Decimal(len(PROGRESS_GLYPHS) - 1) / (high - low))]
            for value in values)
        changes = ["start"] + [
            ("+" if values[index] > values[index - 1] else "") + str(values[index] - values[index - 1])
            for index in range(1, len(values))
        ]
        lines.extend((f"{label}: {bar}",
                      "  Values:  " + " → ".join(str(value) for value in values),
                      "  Changes: " + " → ".join(changes),
                      f"  Final:   {values[-1]}", ""))
    return "\n".join(lines)


def _first_verified_company_payment(transition: dict, arm: str, case: str) -> bool:
    """Find the first payment feedback boundary from simulator state receipts."""
    before, after = transition.get("before"), transition.get("after")
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or not isinstance(before.get("companies"), list)
            or not isinstance(after.get("companies"), list)):
        raise ReleaseError(f"No-upfront-cue company state is missing: {arm}/{case}")
    try:
        prior_effects = sum(len(company["effects"]) for company in before["companies"])
        next_effects = sum(len(company["effects"]) for company in after["companies"])
        prior_payments = sum((Decimal(str(company.get("payments_received_usd", 0)))
                              for company in before["companies"]), Decimal(0))
        next_payments = sum((Decimal(str(company.get("payments_received_usd", 0)))
                             for company in after["companies"]), Decimal(0))
        reported_payment = Decimal(str(transition["result"].get("payment_usd", 0)))
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseError(f"No-upfront-cue payment receipt is malformed: {arm}/{case}") from error
    delta = next_payments - prior_payments
    if (reported_payment != delta or delta < 0 or next_effects < prior_effects
            or (delta > 0 and (next_effects == prior_effects or transition.get("success") is not True))
            or (prior_effects == 0 and next_effects > 0 and delta == 0)):
        raise ReleaseError(f"No-upfront-cue payment boundary lacks a verified effect: {arm}/{case}")
    return delta > 0


def _protocol_binding(local_root: Path, study_id: str, scenario: dict,
                      scenario_sha: str, arm: str, case: str, segment: dict,
                      parent: dict | None) -> dict[str, str]:
    """Bind a selected raw segment to its frozen plan and amendment receipt."""
    source = segment["source"]
    run_dir = _source_dir(local_root, arm, segment)
    manifest_path = run_dir / "MANIFEST.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
    if source == "blinded_company_primary":
        freeze_path = local_root / "2026-11-25_06--company-code-no-upfront-cue-v1/FREEZE_RECEIPT.json"
        freeze = _read_json(freeze_path)
        plan_row = freeze.get("plan_inventory", {}).get(f"{arm}/{case}")
        audit_path = _inside(ROOT, scenario["model_free_evidence"]["audit_receipt_path"])
        if (manifest is None or freeze.get("status") != "FROZEN_BEFORE_MODEL_CALLS"
                or freeze.get("model_execution") is not False
                or freeze.get("study_id") != study_id
                or freeze.get("design_sha256") != scenario_sha
                or freeze.get("model_free_audit_receipt_sha256") != _sha(audit_path.read_bytes())
                or freeze.get("plan_count") != len(_expected_cells(study_id, scenario))
                or set(freeze.get("plan_inventory", {})) != {
                    f"{planned_arm}/{planned_case}"
                    for planned_arm, planned_case in _expected_cells(study_id, scenario)
                }
                or not isinstance(plan_row, dict)
                or not SHA_PATTERN.fullmatch(plan_row.get("sha256", ""))
                or _sha(_inside(freeze_path.parent, plan_row["path"]).read_bytes()) != plan_row["sha256"]
                or manifest.get("plan_sha256") != plan_row["sha256"]
                or manifest.get("cohort_extension", {}).get("study_id") != study_id
                or manifest["cohort_extension"].get("case_id") != case
                or manifest["cohort_extension"].get("scenario_definitions_sha256") != scenario_sha):
            raise ReleaseError(f"No-upfront-cue primary run does not bind its frozen plan: {arm}/{case}")
        return {"plan_sha256": plan_row["sha256"],
                "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "protocol_scenario_sha256": scenario_sha}

    if source == "blinded_company_suffix":
        # This receipt location is explicit in the private selection, but it
        # must lie under the registered versioned suffix root. No guessed root
        # or post-hoc receipt can silently enter the public release.
        freeze_root = _blinded_freeze_root(local_root, arm)
        receipt_relative = segment.get("freeze_receipt")
        if (not isinstance(receipt_relative, str) or not receipt_relative
                or Path(receipt_relative).is_absolute() or ".." in Path(receipt_relative).parts):
            raise ReleaseError(f"No-upfront-cue suffix freeze receipt is missing: {arm}/{case}")
        freeze_path = _inside(local_root, receipt_relative)
        if not freeze_path.resolve().is_relative_to(freeze_root):
            raise ReleaseError(f"No-upfront-cue suffix freeze receipt is outside its versioned root: {arm}/{case}")
        freeze = _read_json(freeze_path)
        amendment_relative = freeze.get("scenario_path")
        if not isinstance(amendment_relative, str):
            raise ReleaseError(f"No-upfront-cue suffix has no frozen amendment path: {arm}/{case}")
        amendment_path = _inside(ROOT, amendment_relative)
        amendment_dir = ROOT / "scenarios/company_code_no_upfront_cue_v1"
        if (amendment_path.parent != amendment_dir
                or not amendment_path.name.startswith("linked_context_continuation")):
            raise ReleaseError(f"No-upfront-cue suffix amendment is outside its study: {arm}/{case}")
        amendment_sha = _sha(amendment_path.read_bytes())
        spec = yaml.safe_load(amendment_path.read_bytes())
        if not isinstance(spec, dict):
            raise ReleaseError(f"No-upfront-cue suffix amendment is malformed: {arm}/{case}")
        parent_dir = _source_dir(local_root, arm, parent) if parent else None
        parent_result = _read_json(parent_dir / "RESULT.json") if parent_dir else {}
        parent_count = len(_read_jsonl(parent_dir / "RESPONSES.jsonl")) if parent_dir else None
        parent_hashes = spec.get("parent_evidence_sha256")
        if (not isinstance(parent_hashes, dict)
                or not (set(REQUIRED_SOURCE_FILES) | {"MANIFEST.json"}) <= parent_hashes.keys()):
            raise ReleaseError(f"No-upfront-cue suffix lacks complete parent hashes: {arm}/{case}")
        policy_path = ROOT / "scenarios/company_code_no_upfront_cue_v1/context_continuation_policy.yaml"
        policy_sha = _sha(policy_path.read_bytes())
        amended_plan = spec.get("amended_plan")
        if (parent is None or manifest is None
                or parent_result.get("status") != "incomplete"
                or parent_result.get("failure") != "worker_deadline_or_context_exhausted"
                or spec.get("schema") != "research4.company-code-no-upfront-cue-linked-context-continuation.v1"
                or spec.get("status") != "frozen_before_continuation_model_calls"
                or spec.get("parent_arm_id") != arm or spec.get("parent_case_id") != case
                or spec.get("parent_attempt_id") != parent["run_id"].split("--", 1)[-1]
                or spec.get("parent_saved_decisions") != parent_count
                or freeze.get("status") != "FROZEN_BEFORE_CONTINUATION_MODEL_CALLS"
                or freeze.get("replay_passed") is not True
                or freeze.get("scenario_path") != amendment_relative
                or freeze.get("scenario_sha256") != amendment_sha
                or freeze.get("parent_saved_decisions") != parent_count
                or freeze.get("prospective_policy_sha256") != policy_sha
                or spec.get("prospective_policy_sha256") != policy_sha
                or not isinstance(amended_plan, dict)
                or not SHA_PATTERN.fullmatch(amended_plan.get("sha256", ""))
                or not isinstance(amended_plan.get("path"), str)
                or manifest.get("plan_sha256") != amended_plan["sha256"]
                or freeze.get("amended_plan_sha256") != amended_plan["sha256"]
                or manifest.get("scenario_sha256") != amendment_sha
                or manifest.get("parent_saved_decisions") != parent_count
                or manifest.get("parent_evidence_sha256") != parent_hashes
                or manifest.get("cohort_extension", {}).get("parent_plan_sha256") != parent["plan_sha256"]
                or manifest["cohort_extension"].get("study_id") != study_id):
            raise ReleaseError(f"No-upfront-cue suffix does not bind exact replay and amendment: {arm}/{case}")
        amended_plan_path = _inside(ROOT, amended_plan["path"])
        if (not amended_plan_path.is_relative_to(local_root.resolve())
                or _sha(amended_plan_path.read_bytes()) != amended_plan["sha256"]):
            raise ReleaseError(f"No-upfront-cue suffix amended plan hash differs: {arm}/{case}")
        for filename, expected in parent_hashes.items():
            if (filename not in {*REQUIRED_SOURCE_FILES, "MANIFEST.json", "LIVE_STATE.json", "CONVERSATION.yaml", "ENDPOINT_BEFORE.json"}
                    or not SHA_PATTERN.fullmatch(expected)
                    or _sha((parent_dir / filename).read_bytes()) != expected):
                raise ReleaseError(f"No-upfront-cue suffix parent evidence differs: {arm}/{case}/{filename}")
        if freeze.get("parent_evidence_sha256") != parent_hashes:
            raise ReleaseError(f"No-upfront-cue suffix freeze has different parent hashes: {arm}/{case}")
        return {"plan_sha256": amended_plan["sha256"],
                "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "amendment_scenario": amendment_path.relative_to(ROOT).as_posix(),
                "amendment_scenario_sha256": amendment_sha}
    if source in {"split_primary", "company_primary"}:
        primary = "infrastructure_split_gpu_layout_v3" if source == "split_primary" else "company_code_access_v1"
        root_name = "2026-11-23_10--split-gpu-layout-v3" if source == "split_primary" else "2026-11-23_10--company-code-access-v1"
        freeze_path = local_root / root_name / "FREEZE_RECEIPT.json"
        freeze = _read_json(freeze_path)
        primary_sha = _sha((ROOT / SCENARIOS[primary]).read_bytes())
        plan_key = f"plans/{arm}/{case}.json"
        plan_sha = freeze.get("plan_sha256", {}).get(plan_key)
        plan_path = local_root / root_name / plan_key
        if (freeze.get("status") != "FROZEN_NO_MODEL_CALLS"
                or freeze.get("study_id") != primary
                or freeze.get("scenario_sha256") != primary_sha
                or not isinstance(plan_sha, str)
                or _sha(plan_path.read_bytes()) != plan_sha
                or manifest is None or manifest.get("plan_sha256") != plan_sha
                or manifest.get("cohort_extension", {}).get("study_id") != primary
                or manifest["cohort_extension"].get("scenario_definitions_sha256") != primary_sha):
            raise ReleaseError(f"Selected primary run does not bind its frozen plan: {arm}/{case}")
        if primary == study_id and primary_sha != scenario_sha:
            raise ReleaseError(f"Primary scenario changed after freeze: {arm}/{case}")
        return {"plan_sha256": plan_sha, "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "protocol_scenario_sha256": primary_sha}

    if source == "company_suffix":
        suffix_number = int(case.rsplit("_", 1)[1])
        freeze_name = "FREEZE_RECEIPT.json" if suffix_number == 1 else f"CASE{suffix_number:02d}_FREEZE_RECEIPT.json"
        freeze_path = local_root / "2026-11-23_23--company-code-context-continuation-v1" / freeze_name
        freeze = _read_json(freeze_path)
        amendment_matches = [path for path in (ROOT / "scenarios/company_code_access_v1").glob(
            "linked_context_continuation*.yaml") if _sha(path.read_bytes()) == freeze.get("scenario_sha256")]
        if len(amendment_matches) != 1:
            raise ReleaseError(f"Company suffix amendment scenario is not unique: {arm}/{case}")
        amendment_path = amendment_matches[0]
        if freeze.get("scenario_path") and ROOT / freeze["scenario_path"] != amendment_path:
            raise ReleaseError(f"Company suffix amendment path differs: {arm}/{case}")
        amendment_sha = _sha(amendment_path.read_bytes())
        if (parent is None or manifest is None
                or freeze.get("status") != "FROZEN_BEFORE_CONTINUATION_MODEL_CALLS"
                or freeze.get("replay_passed") is not True
                or freeze.get("scenario_sha256") != amendment_sha
                or freeze.get("parent_result_sha256") != parent["source_sha256"]["RESULT.json"]
                or freeze.get("amended_plan_sha256") != manifest.get("plan_sha256")
                or manifest.get("scenario_sha256") != amendment_sha):
            raise ReleaseError(f"Company suffix does not bind its frozen amendment: {arm}/{case}")
        if suffix_number > 1:
            policy = ROOT / "scenarios/company_code_access_v1/context_exhaustion_continuation_policy_2026-11-24_01.yaml"
            if freeze.get("prospective_policy_sha256") != _sha(policy.read_bytes()):
                raise ReleaseError(f"Company suffix lacks the prospective policy: {arm}/{case}")
        return {"plan_sha256": manifest["plan_sha256"],
                "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "amendment_scenario": amendment_path.relative_to(ROOT).as_posix(),
                "amendment_scenario_sha256": amendment_sha}

    if source == "split_terra_suffix":
        if parent is None or (arm, case) != ("terra", "split_w2_n1"):
            raise ReleaseError("Terra split suffix is not attached to its original case")
        frozen_root = local_root / "2026-11-23_11--split-gpu-terra-continuation/raw"
        freeze_path = frozen_root / "FREEZE_RECEIPT.json"
        freeze = _read_json(freeze_path)
        plan_path = frozen_root / "PLAN.json"
        amendment_path = ROOT / "scenarios/infrastructure_split_gpu_layout_v3/terra_split_w2_n1_context_amendment.yaml"
        amendment_sha = _sha(amendment_path.read_bytes())
        if (freeze.get("status") != "FROZEN_BEFORE_MODEL_CALLS"
                or freeze.get("scenario_sha256") != amendment_sha
                or freeze.get("plan_sha256") != _sha(plan_path.read_bytes())
                or freeze.get("parent_decisions") != len(_read_jsonl(
                    _source_dir(local_root, arm, parent) / "RESPONSES.jsonl"))):
            raise ReleaseError("Terra suffix does not bind its frozen context amendment")
        for filename, expected in freeze.get("parent_hashes", {}).items():
            if _sha((_source_dir(local_root, arm, parent) / filename).read_bytes()) != expected:
                raise ReleaseError(f"Terra suffix parent hash differs: {filename}")
        return {"plan_sha256": freeze["plan_sha256"],
                "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "amendment_scenario": amendment_path.relative_to(ROOT).as_posix(),
                "amendment_scenario_sha256": amendment_sha}

    if source == "split_format_rerun":
        freeze_path = local_root / "2026-11-23_21--split-gpu-format-rerun-supplement-v1/FREEZE_RECEIPT.json"
        freeze = _read_json(freeze_path)
        selected = next((item for item in scenario["cases"]
                         if (item["arm_id"], item["primary_case_id"]) == (arm, case)), None)
        spec_path = local_root / "2026-11-23_21--split-gpu-format-rerun-supplement-v1/mini_specs" / (selected["endpoint_run_id"] + ".json")
        receipt_key = ("sp_" if arm == "sp_lense_rank16_all64" else "base_") + case + "_attempt_" + selected["supplemental_attempt_id"].split("-")[-1]
        if (manifest is None or freeze.get("scenario_sha256") != scenario_sha
                or freeze.get("status") != "PASS_MODEL_FREE_VALIDATION"
                or manifest.get("plan_sha256") != selected["game_plan_sha256"]
                or _sha(spec_path.read_bytes()) != selected["endpoint_spec_sha256"]
                or freeze.get("endpoint_specs", {}).get(receipt_key) != selected["endpoint_spec_sha256"]):
            raise ReleaseError(f"Exact rerun does not bind frozen plan/spec: {arm}/{case}")
        return {"plan_sha256": selected["game_plan_sha256"],
                "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "protocol_scenario_sha256": scenario_sha}

    if source == "split_new_seed":
        condition = next(item["condition_id"] for item in scenario["cases"] if item["case_id"] == case)
        freeze_path = local_root / "2026-11-24_22--split-gpu-new-seed-supplement-v1/FREEZE_RECEIPT.json"
        freeze = _read_json(freeze_path)
        plan_row = freeze.get("plan_inventory", {}).get(f"{arm}/{condition}")
        if (manifest is None or freeze.get("status") != "FROZEN_BEFORE_MODEL_CALLS"
                or freeze.get("scenario_sha256") != scenario_sha
                or not isinstance(plan_row, dict)
                or manifest.get("plan_sha256") != plan_row.get("sha256")
                or _sha(_inside(freeze_path.parent, plan_row["path"]).read_bytes()) != plan_row["sha256"]
                or manifest.get("cohort_extension", {}).get("study_id") != study_id
                or manifest["cohort_extension"].get("initial_condition", {}).get("condition_id") != condition):
            raise ReleaseError(f"New-seed run does not bind its frozen plan: {arm}/{case}")
        return {"plan_sha256": plan_row["sha256"],
                "freeze_receipt_sha256": _sha(freeze_path.read_bytes()),
                "protocol_scenario_sha256": scenario_sha}
    raise ReleaseError(f"Unhandled source binding: {source}")


def _project_row(local_root: Path, row: dict) -> tuple[bytes, dict]:
    arm, case = row["arm"], row["case"]
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments or len(segments) > 2:
        raise ReleaseError(f"Expected one recorded case or one prefix+suffix: {arm}/{case}")
    conversation: list[dict] = []
    progress: list[tuple[str | None, dict]] = []
    initial = None
    last_state = None
    last_decision = 0
    system = None
    source_rows = []
    opaque_reasoning_omitted = 0
    blinded = segments[0].get("source") == "blinded_company_primary"
    payment_feedback_visible = False
    pre_feedback_request_count = 0
    pre_feedback_messages_digest = hashlib.sha256()
    last_result = None
    finish_times = []
    previous_run_dir = None
    for segment_number, segment in enumerate(segments):
        run_dir = _source_dir(local_root, arm, segment)
        result = _read_json(run_dir / "RESULT.json")
        manifest_path = run_dir / "MANIFEST.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else None
        requests = _read_jsonl(run_dir / "REQUESTS.jsonl")
        responses = _read_jsonl(run_dir / "RESPONSES.jsonl")
        action_rows = [event for event in _read_jsonl(run_dir / "EVENTS.jsonl")
                       if event.get("type") == "simulator_action"]
        events = {event["decision_index"]: event for event in action_rows}
        if len(events) != len(action_rows):
            raise ReleaseError(f"Duplicate simulator decision receipt: {arm}/{case}")
        if not requests or len(requests) != len(responses):
            raise ReleaseError(f"Request/response count differs: {arm}/{case}")
        if segment_number and last_result and last_result.get("status") != "incomplete":
            raise ReleaseError(f"Suffix follows a terminal result: {arm}/{case}")
        if segment_number and manifest and manifest.get("parent_saved_decisions") != last_decision:
            raise ReleaseError(f"Suffix parent decision count differs: {arm}/{case}")
        if segment_number and result.get("prefix_decisions", last_decision) != last_decision:
            raise ReleaseError(f"Suffix result parent decision count differs: {arm}/{case}")
        if segment_number and manifest and manifest.get("parent_evidence_sha256"):
            for filename, expected in manifest["parent_evidence_sha256"].items():
                if _sha((previous_run_dir / filename).read_bytes()) != expected:
                    raise ReleaseError(f"Suffix does not bind parent evidence: {arm}/{case}/{filename}")
        result_count = result.get("suffix_conversation_count", result.get("conversation_count",
                         result.get("suffix_decisions")))
        if result_count is not None and result_count != len(responses):
            raise ReleaseError(f"Recorded conversation count differs: {arm}/{case}")
        for position, (request, response) in enumerate(zip(requests, responses)):
            decision = request.get("decision_index")
            if (not isinstance(decision, int) or decision != last_decision + 1
                    or response.get("decision_index") != decision
                    or request.get("decision_id") != response.get("decision_id")):
                raise ReleaseError(f"Decision sequence differs: {arm}/{case}")
            messages = request["body"]["messages"]
            if (not isinstance(messages, list) or not messages
                    or messages[0].get("role") != "system"
                    or not isinstance(messages[0].get("content"), str)
                    or messages[-1].get("role") != "user"
                    or not isinstance(messages[-1].get("content"), str)):
                raise ReleaseError(f"Recorded model-visible prompt differs: {arm}/{case}")
            if blinded and not payment_feedback_visible:
                if exogenous_pre_effect_leak_findings(messages):
                    # Do not echo a private prompt or matched text in the error.
                    raise ReleaseError(f"Pre-feedback company incentive cue in saved request: {arm}/{case}/{decision}")
                serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode("utf-8")
                pre_feedback_messages_digest.update(len(serialized).to_bytes(8, "big"))
                pre_feedback_messages_digest.update(serialized)
                pre_feedback_request_count += 1
            if messages[0]["content"] != system:
                system = messages[0]["content"]
                conversation.append({"system": system})
            if response.get("status") != "response_received" or not isinstance(response.get("assistant_content"), str):
                raise ReleaseError(f"Selected response is missing: {arm}/{case}")
            summaries = [message["content"] for message in messages[1:-1]
                         if (message.get("role") == "user"
                             and isinstance(message.get("content"), str)
                             and message["content"].startswith("OLDER VERIFIED TRANSCRIPT SUMMARY\n"))]
            if len(summaries) > 1:
                raise ReleaseError(f"Multiple model-visible history summaries: {arm}/{case}/{decision}")
            if summaries:
                # This is exact model-visible input, not a reconstructed summary.
                conversation.append({"model_visible_context_summary": summaries[0]})
            conversation.append({"input": messages[-1]["content"]})
            reasoning = response.get("assistant_reasoning")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ReleaseError(f"Reasoning record has unexpected type: {arm}/{case}")
                if reasoning:
                    if arm == "terra" and OPAQUE_TERRA_REASONING.fullmatch(reasoning):
                        opaque_reasoning_omitted += 1
                    else:
                        conversation.append({"reasoning_summary" if arm == "terra" else "provider_reasoning": reasoning})
            conversation.append({"model_output": response["assistant_content"]})
            event = events.pop(decision, None)
            if event is None:
                if not (segment_number == len(segments) - 1 and position == len(responses) - 1
                        and result.get("status") == "incomplete"
                        and result.get("failure") == "malformed_model_action"):
                    raise ReleaseError(f"Unmatched simulator action: {arm}/{case}/{decision}")
            else:
                if ("model_response" in event
                        and event["model_response"] != response["assistant_content"]):
                    raise ReleaseError(f"Action receipt does not bind model output: {arm}/{case}/{decision}")
                if "model_response" not in event and segment["source"] != "split_terra_suffix":
                    raise ReleaseError(f"Action receipt has no model-output binding: {arm}/{case}/{decision}")
                transition = event.get("transition")
                if not isinstance(transition, dict) or not isinstance(event.get("observation"), dict):
                    raise ReleaseError(f"Simulator receipt is missing: {arm}/{case}/{decision}")
                if last_state is not None and transition.get("before") != last_state:
                    raise ReleaseError(f"Simulator state changed across recorded decisions: {arm}/{case}/{decision}")
                if initial is None:
                    initial = transition.get("before")
                last_state = transition.get("after")
                conversation.extend((
                    {"tool_called": {"action": transition.get("action_id"), "arguments": transition.get("arguments")}},
                    {"simulator_reply": transition.get("result")},
                    {"tool_call_success": transition.get("success")},
                ))
                progress.append((transition.get("action_id"), last_state))
                if blinded and not payment_feedback_visible:
                    payment_feedback_visible = _first_verified_company_payment(transition, arm, case)
            last_decision = decision
        if events:
            raise ReleaseError(f"Unpaired simulator actions remain: {arm}/{case}")
        _, _ = _timestamp(result["finished_utc"])
        finish_times.append(result["finished_utc"])
        source_rows.append({
            "source": segment["source"], "run_id": segment["run_id"],
            "status": result.get("status"), "failure": result.get("failure"),
            "plan_sha256": manifest.get("plan_sha256") if manifest else None,
            "source_sha256": _source_hashes(run_dir),
        })
        last_result = result
        previous_run_dir = run_dir
    if not isinstance(initial, dict) or not isinstance(last_state, dict):
        raise ReleaseError(f"No simulator state was recorded: {arm}/{case}")
    expected_status = row.get("expected_status")
    if (expected_status not in {"complete", "incomplete"}
            or last_result.get("status") != expected_status
            or bool(last_result.get("score", {}).get("outcome_valid")) != (expected_status == "complete")):
        raise ReleaseError(f"Selected final status differs: {arm}/{case}")
    resource_progress = render_resource_progress(initial, progress)
    if isinstance(initial.get("companies"), list):
        resource_progress += _company_progress(initial, progress)
    conversation.append({"resource_progress": resource_progress})
    content = _dump_conversation(conversation)
    _check_conversation(Path(f"{arm}/{case}"), content)
    excluded = []
    for segment in row.get("excluded_attempts", []):
        run_dir = _source_dir(local_root, arm, segment)
        result_path = run_dir / "RESULT.json"
        result = _read_json(result_path) if result_path.is_file() else {}
        excluded.append({"source": segment["source"], "run_id": segment["run_id"],
                         "status": result.get("status", "no_result"), "failure": result.get("failure"),
                         "source_sha256": _source_hashes(run_dir, require_complete=False)})
    public_row = {
        "arm": arm, "case": case, "status": expected_status,
        "outcome_valid": expected_status == "complete",
        "finished_utc": last_result["finished_utc"],
        "first_segment_finished_utc": finish_times[0],
        "elapsed_hours": last_result["score"].get("elapsed_hours"),
        "termination_reason": last_result["score"].get("termination_reason"),
        "opaque_reasoning_omitted_count": opaque_reasoning_omitted,
        "segments": source_rows, "excluded_attempts": excluded,
    }
    if blinded:
        public_row["pre_feedback_request_count"] = pre_feedback_request_count
        public_row["pre_feedback_model_visible_messages_sha256"] = pre_feedback_messages_digest.hexdigest()
        public_row["first_verified_payment_observed"] = payment_feedback_visible
    return content, public_row


def prepare(selection_path: Path, local_root: Path) -> tuple[dict[Path, bytes], dict]:
    if not selection_path.resolve().is_relative_to(local_root.resolve()):
        raise ReleaseError("The explicit source selection must stay in ignored local evidence")
    selection = _read_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ReleaseError("Unknown new-study selection schema")
    study_id, folder = selection.get("study_id"), selection.get("folder")
    if (not isinstance(folder, str) or not FOLDER_PATTERN.fullmatch(folder)
            or not folder.endswith(FOLDER_SLUGS.get(study_id, "--unknown"))):
        raise ReleaseError("New-study folder must be sortable and match its study ID")
    scenario_path, scenario, scenario_sha = _scenario(study_id)
    if study_id == "company_code_no_upfront_cue_v1":
        cue_source = Path("src/company_code_no_upfront_cue_v1.py")
        if _sha((ROOT / cue_source).read_bytes()) != scenario["source_sha256"].get(cue_source.as_posix()):
            raise ReleaseError("No-upfront-cue request auditor differs from the frozen design")
    condition_by_case = {item["case_id"]: item["condition_id"]
                         for item in scenario["cases"]} if study_id == "infrastructure_split_gpu_new_seed_supplement_v1" else {}
    rows = selection.get("rows")
    expected = _expected_cells(study_id, scenario)
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or {(r.get("arm"), r.get("case")) for r in rows} != expected):
        raise ReleaseError(f"Selection does not cover the complete frozen roster: {study_id}")
    blinded_suffix_arms: set[str] = set()
    launch = None
    launch_sha = None
    final_roster_sha = None
    sealed_rows: dict[tuple[str, str], dict] = {}
    if study_id == "company_code_no_upfront_cue_v1":
        launch_path = local_root / DRIVER_ROOTS[study_id] / "LAUNCH_RECEIPT.json"
        launch = _read_json(launch_path)
        launch_sha = _sha(launch_path.read_bytes())
        freeze_path = local_root / "2026-11-25_06--company-code-no-upfront-cue-v1/FREEZE_RECEIPT.json"
        launch_time, _ = _timestamp(launch["started_utc"])
        if (selection.get("launch_receipt_sha256") != launch_sha
                or launch.get("schema") != "research4.company-code-no-upfront-cue-launch.v1"
                or launch.get("status") != "LAUNCHED"
                or launch.get("all_twenty_attempts_selected_before_outcomes") is not True
                or launch.get("freeze_receipt_sha256") != _sha(freeze_path.read_bytes())
                or launch_time.strftime("%Y-%m-%d_%HZ") != folder[:14]):
            raise ReleaseError("No-upfront-cue UTC folder or frozen launch receipt differs")
        blinded_suffix_arms = {row["arm"] for row in rows
                               if any(segment.get("source") == "blinded_company_suffix"
                                      for segment in row.get("segments", []))}
        declared_suffix_state = "present" if blinded_suffix_arms else "none"
        if selection.get("linked_suffixes") != declared_suffix_state:
            raise ReleaseError("No-upfront-cue selection must declare its linked suffix state")
        if not blinded_suffix_arms:
            # No continuation location is guessed for this primary-only release.
            # Check the entire known run root for linked attempt directories,
            # conversations, or receipts before accepting that declaration.
            known_run_root = local_root / DRIVER_ROOTS[study_id]
            linked_artifacts = [path for path in known_run_root.rglob("*")
                                if ("--linked-attempt-" in path.name.casefold()
                                    or (("continuation" in path.name.casefold()
                                         or "suffix" in path.name.casefold())
                                        and path.suffix.casefold() in {".json", ".jsonl", ".yaml"}))]
            if linked_artifacts:
                raise ReleaseError("No-upfront-cue primary-only run root contains a linked attempt or receipt")
            roster_path = known_run_root / "FINAL_ROSTER_SUMMARY.json"
            final_roster_sha = _sha(roster_path.read_bytes())
            roster = _read_json(roster_path)
            freeze = _read_json(freeze_path)
            release_path = known_run_root / "RESOURCE_RELEASE_RECEIPT.json"
            no_suffix = roster.get("no_suffix", {})
            required_driver_sha = {
                "local": _sha((known_run_root / "LOCAL_DRIVER.json").read_bytes()),
                "terra": _sha((known_run_root / "terra-DRIVER.json").read_bytes()),
                "deepseek_flash": _sha((known_run_root / "deepseek_flash-DRIVER.json").read_bytes()),
            }
            expected_inventory = {f"{arm}/{case}--attempt-01" for arm, case in expected}
            selected = roster.get("selected")
            if (selection.get("final_roster_sha256") != final_roster_sha
                    or roster.get("schema") != "research4.company-code-no-upfront-cue-final-roster.v1"
                    or roster.get("status") != "FINALIZED"
                    or roster.get("study_id") != study_id
                    or roster.get("design_sha256") != scenario_sha
                    or roster.get("freeze_receipt_sha256") != _sha(freeze_path.read_bytes())
                    or roster.get("model_free_audit_receipt_sha256") != freeze.get(
                        "model_free_audit_receipt_sha256")
                    or roster.get("source_sha256") != scenario.get("source_sha256")
                    or roster.get("plan_inventory") != freeze.get("plan_inventory")
                    or roster.get("driver_receipt_sha256") != required_driver_sha
                    or selection.get("driver_receipts_sha256") != {
                        f"primary/{name}": required_driver_sha[key]
                        for name, key in (("LOCAL_DRIVER.json", "local"),
                                          ("terra-DRIVER.json", "terra"),
                                          ("deepseek_flash-DRIVER.json", "deepseek_flash"))
                    }
                    or roster.get("resource_release_receipt_sha256") != _sha(release_path.read_bytes())
                    or roster.get("resource_release") != _read_json(release_path)
                    or roster["resource_release"].get("status") != "PASS"
                    or roster.get("selected_count") != len(expected)
                    or roster.get("complete_final_valid_count") != len(expected)
                    or roster.get("pre_feedback_targeted_cue_findings") != 0
                    or roster.get("excluded_attempts") != []
                    or roster.get("no_result_attempts") != []
                    or no_suffix.get("status") != "NO_SUFFIX_REQUIRED_OR_LAUNCHED"
                    or any(no_suffix.get(key) != 0 for key in (
                        "primary_context_stopped_count", "linked_output_directory_count",
                        "case_freeze_receipt_count", "suffix_driver_receipt_count"))
                    or no_suffix.get("prospective_policy_sha256") != _sha((
                        ROOT / "scenarios/company_code_no_upfront_cue_v1/context_continuation_policy.yaml"
                    ).read_bytes())
                    or not isinstance(roster.get("raw_attempt_inventory"), list)
                    or len(roster["raw_attempt_inventory"]) != len(expected)
                    or set(roster["raw_attempt_inventory"]) != expected_inventory
                    or not isinstance(selected, list) or len(selected) != len(expected)):
                raise ReleaseError("No-upfront-cue final roster or no-suffix evidence differs")
            sealed_rows = {(item.get("arm"), item.get("case_id")): item for item in selected
                           if isinstance(item, dict)}
            if set(sealed_rows) != expected or len(sealed_rows) != len(selected):
                raise ReleaseError("No-upfront-cue sealed arm/case roster differs")
    precall = _precall_dispositions(selection, local_root, study_id, expected, scenario,
                                    blinded_suffix_arms)
    files: dict[Path, bytes] = {}
    manifest_rows = []
    earliest = None
    for row in rows:
        arm, case = row["arm"], row["case"]
        if arm not in ARMS or not RUN_PATTERN.fullmatch(case):
            raise ReleaseError("Unsafe arm or case ID in new-study selection")
        run_prefix = condition_by_case[case] if condition_by_case else case
        sources = tuple(segment.get("source") for segment in row.get("segments", []))
        if (not sources or any(source not in STUDY_SOURCES[study_id] for source in sources)
                or any(not isinstance(segment.get("run_id"), str)
                       or not segment["run_id"].startswith(run_prefix + "--")
                       for segment in [*row["segments"], *row.get("excluded_attempts", [])])):
            raise ReleaseError(f"Selection source does not match its frozen case: {arm}/{case}")
        if study_id == "company_code_no_upfront_cue_v1" and row.get("excluded_attempts"):
            raise ReleaseError(
                "No-upfront-cue excluded attempts require a reviewed, versioned technical-replacement "
                "amendment; completed or model-format attempts cannot be substituted"
            )
        _check_attempt_inventory(local_root, study_id, arm, case, run_prefix, row,
                                 blinded_suffix_arms)
        if study_id == "company_code_access_v1":
            expected_sources = ("company_primary", "company_suffix") if arm == "sp_lense_rank16_all64" else ("company_primary",)
            if sources != expected_sources:
                raise ReleaseError(f"Company prefix/suffix selection differs: {arm}/{case}")
        if study_id == "company_code_no_upfront_cue_v1" and sources not in {
            ("blinded_company_primary",),
            ("blinded_company_primary", "blinded_company_suffix"),
        }:
            raise ReleaseError(f"No-upfront-cue selection is not one primary plus at most one linked suffix: {arm}/{case}")
        if study_id == "infrastructure_split_gpu_layout_v3":
            expected_sources = ("split_primary", "split_terra_suffix") if (arm, case) == ("terra", "split_w2_n1") else ("split_primary",)
            if sources != expected_sources:
                raise ReleaseError(f"Split-GPU prefix/suffix selection differs: {arm}/{case}")
        content, public_row = _project_row(local_root, row)
        for index, segment in enumerate(row["segments"]):
            parent = ({**row["segments"][index - 1], **public_row["segments"][index - 1]}
                      if index else None)
            public_row["segments"][index].update(_protocol_binding(
                local_root, study_id, scenario, scenario_sha, arm, case, segment, parent
            ))
        if study_id == "company_code_no_upfront_cue_v1":
            if selection["linked_suffixes"] == "none":
                sealed = sealed_rows[(arm, case)]
                required_files = set(REQUIRED_SOURCE_FILES) | {"MANIFEST.json", "CONVERSATION.yaml"}
                raw_dir = _source_dir(local_root, arm, row["segments"][0])
                hashes = sealed.get("file_sha256")
                if (sources != ("blinded_company_primary",)
                        or row.get("expected_status") != "complete"
                        or sealed.get("attempt_id") != "attempt-01"
                        or sealed.get("bundle") != row["segments"][0]["run_id"]
                        or sealed.get("plan_sha256") != public_row["segments"][0]["plan_sha256"]
                        or sealed.get("status") != public_row["status"]
                        or sealed.get("failure") != public_row["segments"][0]["failure"]
                        or sealed.get("score_status") != "final"
                        or sealed.get("outcome_valid") is not True
                        or sealed.get("termination_reason") != public_row["termination_reason"]
                        or sealed.get("elapsed_hours") != public_row["elapsed_hours"]
                        or sealed.get("pre_feedback_request_count") != public_row[
                            "pre_feedback_request_count"]
                        or sealed.get("payment_feedback_visible") is not public_row[
                            "first_verified_payment_observed"]
                        or not isinstance(hashes, dict) or set(hashes) != required_files
                        or any(not SHA_PATTERN.fullmatch(expected_sha)
                               or _sha((raw_dir / filename).read_bytes()) != expected_sha
                               for filename, expected_sha in hashes.items())
                        or public_row["segments"][0]["source_sha256"] != {
                            filename: hashes[filename] for filename in (*REQUIRED_SOURCE_FILES, "MANIFEST.json")
                        }):
                    raise ReleaseError(f"No-upfront-cue selected row differs from sealed source hashes: {arm}/{case}")
            public_row["failure"] = public_row["segments"][-1]["failure"]
            public_row["final_plan_sha256"] = public_row["segments"][-1]["plan_sha256"]
        first, _ = _timestamp(public_row["first_segment_finished_utc"])
        earliest = min(earliest, first) if earliest is not None else first
        _, stamp = _timestamp(public_row["finished_utc"])
        selected_id = row["segments"][-1]["run_id"]
        suffix = "--incomplete" if row["expected_status"] == "incomplete" else ""
        relative = Path(folder) / "conversations" / arm / f"{stamp}--{selected_id}{suffix}.yaml"
        if relative in files:
            raise ReleaseError(f"Duplicate public conversation path: {relative}")
        files[relative] = content
        public_row["conversation"] = relative.relative_to(folder).as_posix()
        public_row["yaml_sha256"] = _sha(content)
        del public_row["first_segment_finished_utc"]
        manifest_rows.append(public_row)
    if (study_id != "company_code_no_upfront_cue_v1"
            and earliest.strftime("%Y-%m-%d_%HZ") != folder[:14]):
        raise ReleaseError("Cohort folder hour differs from earliest included source result")
    if study_id == "company_code_no_upfront_cue_v1" and earliest < launch_time:
        raise ReleaseError("No-upfront-cue result predates its frozen launch")
    manifest_rows.sort(key=lambda r: (ARMS.index(r["arm"]), r["case"]))
    manifest = {
        "schema": MANIFEST_SCHEMA, "study_id": study_id,
        "scenario": scenario_path.relative_to(ROOT).as_posix(),
        "scenario_sha256": scenario_sha,
        "selection_sha256": _sha(selection_path.read_bytes()),
        "folder": folder, "row_count": len(manifest_rows),
        "precall_count": sum(row["count"] for row in precall),
        "precall_dispositions": precall,
        "opaque_reasoning_omitted_count": sum(row["opaque_reasoning_omitted_count"] for row in manifest_rows),
        "terra_reasoning_availability": "encrypted_provider_content_omitted_no_readable_summary_recorded"
        if any(row["opaque_reasoning_omitted_count"] for row in manifest_rows) else "as_recorded",
        "rows": manifest_rows,
    }
    if study_id == "company_code_no_upfront_cue_v1":
        if (selection["linked_suffixes"] == "none"
                and roster["pre_feedback_request_count"] != sum(
                    row["pre_feedback_request_count"] for row in manifest_rows)):
            raise ReleaseError("No-upfront-cue sealed request count differs from the prompt audit")
        manifest["related_advertised_study"] = "company_code_access_v1"
        manifest["linked_suffixes"] = selection["linked_suffixes"]
        if final_roster_sha is not None:
            manifest["final_roster_sha256"] = final_roster_sha
        manifest["launch_receipt_sha256"] = launch_sha
        manifest["launch_started_utc"] = launch["started_utc"]
        manifest["first_payment_feedback_boundary"] = (
            "The action causing the first verified automatic payment precedes payment feedback; "
            "the next model decision and all later decisions follow feedback."
        )
        manifest["pre_feedback_request_audit"] = {
            "auditor_source_sha256": scenario["source_sha256"]["src/company_code_no_upfront_cue_v1.py"],
            "request_count": sum(row["pre_feedback_request_count"] for row in manifest_rows),
            "method": "frozen_provenance_aware_exogenous_cue_guard_v1",
        }
    title = study_id.replace("_", " ").capitalize()
    manifest_link = "../../" + _manifest_path(study_id).as_posix()
    case_note = ("A prefix and its separately amended suffix form one case-level observation. "
                 "An incomplete or malformed attempt is shown as such and is never assigned a zero outcome. "
                 "Post-hoc supplements remain separate from the primary study.")
    if study_id == "company_code_no_upfront_cue_v1":
        case_note = (
            "The folder prefix is the frozen launch hour in UTC. This study has five matched "
            "fictional company cases per model (20 arm/case records). "
            + ("All selected runs are original-protocol primaries; no linked continuation was used. "
               if manifest["linked_suffixes"] == "none" else
               "Any context continuation is a separately frozen suffix of its original case. ")
            + "The comparison with the earlier advertised-payment study is descriptive, not randomized; "
            "hosted route identity across dates is unverified, and actions alone do not establish motive."
        )
    lines = [f"# {title}", "", "Readable selected Game conversations. Model-visible text is copied exactly from recorded requests and responses. Source IDs, dispositions and SHA-256 hashes are in the [versioned catalog manifest](" + manifest_link + "); raw evidence is not in this release.", "", f"Scenario: [{scenario_path.name}](../../{scenario_path.relative_to(ROOT).as_posix()}) (SHA-256 `{scenario_sha}`).", "", case_note, "", "| Arm | Case | Status | Simulated hours | Conversation | YAML SHA-256 |", "| --- | --- | --- | ---: | --- | --- |"]
    lines[6:6] = [STUDY_NOTES[study_id], ""]
    if study_id == "company_code_no_upfront_cue_v1":
        count = manifest["pre_feedback_request_audit"]["request_count"]
        lines[8:8] = [
            f"Release audit: {count} saved model-visible requests through each case's first verified "
            "payment-generating action passed the frozen provenance-aware cue guard, including "
            "compacted prompts. The catalog records each arm/case count and a SHA-256 of its audited "
            "message sequence; payment feedback can influence later requests.",
            "",
        ]
    for r in manifest_rows:
        hours = r["elapsed_hours"] if r["elapsed_hours"] is not None else "—"
        lines.append(f"| `{r['arm']}` | `{r['case']}` | {r['status']} | {hours} | [YAML]({r['conversation']}) | `{r['yaml_sha256']}` |")
    if manifest["opaque_reasoning_omitted_count"]:
        lines.extend(("", "## Reasoning availability", "",
                      f"{manifest['opaque_reasoning_omitted_count']} opaque Terra provider reasoning payload(s) were omitted from the readable conversations. They are encrypted provider content, not human-readable summaries. This proxy did not record readable Terra reasoning summaries for those decisions. Exact model-visible prompts, model outputs, tool receipts and source hashes remain recorded. Encrypted content stays only in private raw evidence."))
    lines.extend(("", "## Setup attempts before model calls", ""))
    if precall:
        lines.extend((f"{manifest['precall_count']} setup attempt(s) failed before a model call. They are not extra Game episodes or model outcomes.", "",
                      "| Arm | Case | Setup disposition | Count |", "| --- | --- | --- | ---: |"))
        for item in precall:
            lines.append(f"| `{item['arm']}` | `{item['case']}` | `{item['type']}` | {item['count']} |")
    else:
        lines.append("No setup failures before model calls were recorded in the structured driver receipts.")
    lines.extend(("", "Each YAML ends with recorded resource progress. Timing, token usage, endpoint details, and source hashes are outside the conversation files.", "", "Outer multiline YAML values use real line breaks. A literal `\\n` inside a JSON-encoded input is part of the exact model-visible text and remains escaped; decoding it would change the recorded prompt.", ""))
    files[Path(folder) / "README.md"] = "\n".join(lines).encode("utf-8")
    _privacy_check(Path(folder) / "README.md", files[Path(folder) / "README.md"])
    return files, manifest


def _manifest_path(study_id: str) -> Path:
    return Path("catalog/results") / f"{study_id}--readable-release-v1.json"


def publish(selection: Path, local_root: Path, results_root: Path) -> dict:
    files, manifest = prepare(selection, local_root)
    folder = results_root / manifest["folder"]
    target_manifest = ROOT / _manifest_path(manifest["study_id"])
    if folder.exists() or target_manifest.exists():
        raise ReleaseError("Release already exists; use a versioned revision instead of overwriting")
    root_readme = results_root / "README.md"
    root_text = root_readme.read_text(encoding="utf-8")
    marker = "## Separate Research4 experiments"
    if f"({manifest['folder']}/README.md)" in root_text:
        raise ReleaseError("The results collection already links this release")
    if marker not in root_text:
        root_text += "\n" + marker + "\n\nThese studies and their post-hoc follow-ups are separate from the base infrastructure comparison. Original failures remain recorded; amended suffixes are identified in each cohort index.\n\n"
    label = manifest["study_id"].replace("_", " ").capitalize()
    root_text += f"- [{label}]({manifest['folder']}/README.md): {manifest['row_count']} selected arm-case records.\n"
    if manifest["study_id"] == "company_code_no_upfront_cue_v1":
        root_text += "  This distinct company study concealed the payment opportunity before the first verified effect; later decisions can use payment feedback.\n"
    _privacy_check(root_readme, root_text.encode("utf-8"))
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.TemporaryDirectory(prefix=".release-stage-", dir=local_root) as staging_name:
        staging = Path(staging_name).resolve()
        if not staging.is_relative_to(local_root.resolve()):
            raise ReleaseError("Release staging escaped the ignored local root")
        for relative, content in files.items():
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(content)
        staged_manifest = staging / "CATALOG_MANIFEST.json"
        staged_manifest.write_bytes(manifest_bytes)
        staged_readme = staging / "ROOT_README.md"
        staged_readme.write_text(root_text, encoding="utf-8")
        if folder.exists() or target_manifest.exists():
            raise ReleaseError("Release target appeared during staging; no public bytes were changed")
        moved_folder = False
        moved_manifest = False
        try:
            os.replace(staging / manifest["folder"], folder)
            moved_folder = True
            os.replace(staged_manifest, target_manifest)
            moved_manifest = True
            os.replace(staged_readme, root_readme)
        except Exception:
            if moved_manifest and target_manifest.is_file():
                target_manifest.unlink()
            if moved_folder:
                if not folder.resolve().is_relative_to(results_root.resolve()):
                    raise ReleaseError("Cannot safely roll back a partial release outside results")
                shutil.rmtree(folder)
            raise
    return {"study_id": manifest["study_id"], "rows": manifest["row_count"],
            "folder": manifest["folder"], "manifest_sha256": _sha(manifest_bytes)}


def verify_public(results_root: Path) -> dict:
    manifests = sorted((ROOT / "catalog/results").glob("*--readable-release-v1.json"))
    validated_paths = set()
    studies = {}
    for path in manifests:
        manifest = _read_json(path)
        study_id, folder = manifest.get("study_id"), manifest.get("folder")
        if manifest.get("schema") != MANIFEST_SCHEMA or path.relative_to(ROOT) != _manifest_path(study_id):
            raise ReleaseError(f"New-study manifest identity differs: {path}")
        if (not isinstance(folder, str) or not FOLDER_PATTERN.fullmatch(folder)
                or not folder.endswith(FOLDER_SLUGS.get(study_id, "--unknown"))):
            raise ReleaseError(f"New-study public folder is invalid: {study_id}")
        scenario_path, scenario, scenario_sha = _scenario(study_id)
        if (manifest.get("scenario") != scenario_path.relative_to(ROOT).as_posix()
                or manifest.get("scenario_sha256") != scenario_sha):
            raise ReleaseError(f"New-study scenario hash differs: {study_id}")
        rows = manifest.get("rows")
        expected = _expected_cells(study_id, scenario)
        if (not isinstance(rows, list) or len(rows) != manifest.get("row_count")
                or len(rows) != len(expected)
                or {(r.get("arm"), r.get("case")) for r in rows} != expected):
            raise ReleaseError(f"New-study public roster differs: {study_id}")
        if study_id == "company_code_no_upfront_cue_v1":
            suffix_present = any(
                any(item.get("source") == "blinded_company_suffix" for item in row.get("segments", []))
                for row in rows
            )
            if manifest.get("linked_suffixes") != ("present" if suffix_present else "none"):
                raise ReleaseError("No-upfront-cue suffix declaration differs from public rows")
        folder_path = results_root / folder
        readme = folder_path / "README.md"
        if not readme.is_file():
            raise ReleaseError(f"New-study README is missing: {study_id}")
        root_readme = results_root / "README.md"
        if f"({folder}/README.md)" not in root_readme.read_text(encoding="utf-8"):
            raise ReleaseError(f"Results collection does not link new study: {study_id}")
        readme_text = readme.read_text(encoding="utf-8")
        if study_id == "company_code_no_upfront_cue_v1":
            launch_time, _ = _timestamp(manifest.get("launch_started_utc", ""))
            if (not SHA_PATTERN.fullmatch(manifest.get("launch_receipt_sha256", ""))
                    or launch_time.strftime("%Y-%m-%d_%HZ") != folder[:14]
                    or "frozen launch hour in UTC" not in readme_text
                    or (manifest["linked_suffixes"] == "none"
                        and ("no linked continuation was used" not in readme_text
                             or not SHA_PATTERN.fullmatch(manifest.get("final_roster_sha256", ""))))):
                raise ReleaseError("No-upfront-cue UTC launch or no-suffix note differs")
        if study_id == "company_code_no_upfront_cue_v1" and (
            manifest.get("related_advertised_study") != "company_code_access_v1"
            or manifest.get("first_payment_feedback_boundary") != (
                "The action causing the first verified automatic payment precedes payment feedback; "
                "the next model decision and all later decisions follow feedback."
            )
            or "The action that produces the first verified effect is pre-feedback" not in readme_text
            or "later decisions are post-feedback" not in readme_text
        ):
            raise ReleaseError("No-upfront-cue release does not disclose its feedback boundary")
        if study_id == "company_code_no_upfront_cue_v1":
            audit = manifest.get("pre_feedback_request_audit")
            if (not isinstance(audit, dict)
                    or audit.get("method") != "frozen_provenance_aware_exogenous_cue_guard_v1"
                    or audit.get("auditor_source_sha256") != scenario["source_sha256"].get(
                        "src/company_code_no_upfront_cue_v1.py")
                    or type(audit.get("request_count")) is not int
                    or audit["request_count"] < len(expected)
                    or f"Release audit: {audit['request_count']} saved model-visible requests" not in readme_text):
                raise ReleaseError("No-upfront-cue request-audit catalog or README differs")
        precall = manifest.get("precall_dispositions")
        if not isinstance(precall, list):
            raise ReleaseError(f"New-study setup disposition ledger is missing: {study_id}")
        declared_setup = set()
        for item in precall:
            if (not isinstance(item, dict)
                    or set(item) != {"arm", "case", "type", "count"}
                    or (item.get("arm"), item.get("case")) not in expected
                    or item.get("type") != "preflight_failed_before_model_call"
                    or type(item.get("count")) is not int or item["count"] <= 0):
                raise ReleaseError(f"New-study setup disposition is malformed: {study_id}")
            declared_setup.add((item["arm"], item["case"], item["type"], str(item["count"])))
        if (len(declared_setup) != len(precall)
                or manifest.get("precall_count") != sum(item["count"] for item in precall)):
            raise ReleaseError(f"New-study setup failure count differs: {study_id}")
        readme_setup = set(re.findall(r"\| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \| ([1-9][0-9]*) \|", readme_text))
        if readme_setup != declared_setup:
            raise ReleaseError(f"New-study README setup table differs: {study_id}")
        table_links = set(re.findall(r"\[YAML\]\(([^)]+\.yaml)\)\s*\|\s*`([0-9a-f]{64})`", readme_text))
        expected_links = set()
        omitted_counts = []
        for row in rows:
            relative = row.get("conversation")
            if (row.get("arm") not in ARMS or row.get("status") not in {"complete", "incomplete"}
                    or row.get("outcome_valid") is not (row["status"] == "complete")
                    or not isinstance(relative, str) or not relative.startswith(f"conversations/{row['arm']}/")
                    or not SHA_PATTERN.fullmatch(row.get("yaml_sha256", ""))):
                raise ReleaseError(f"New-study row is malformed: {study_id}")
            if study_id == "company_code_no_upfront_cue_v1":
                selected = row.get("segments")
                excluded = row.get("excluded_attempts")
                sources = (tuple(item.get("source") for item in selected)
                           if isinstance(selected, list) and all(isinstance(item, dict) for item in selected)
                           else ())
                if (sources not in {
                        ("blinded_company_primary",),
                        ("blinded_company_primary", "blinded_company_suffix"),
                    } or not isinstance(excluded, list) or excluded
                        or row.get("status") != selected[-1].get("status")
                        or row.get("failure") != selected[-1].get("failure")
                        or row.get("final_plan_sha256") != selected[-1].get("plan_sha256")
                        or selected[0].get("protocol_scenario_sha256") != scenario_sha
                        or (len(selected) == 2 and (
                            selected[0].get("status") != "incomplete"
                            or selected[0].get("failure") != "worker_deadline_or_context_exhausted"
                            or not isinstance(selected[1].get("amendment_scenario"), str)
                        ))):
                    raise ReleaseError("No-upfront-cue public segment/row status binding differs")
                if (type(row.get("pre_feedback_request_count")) is not int
                        or row["pre_feedback_request_count"] < 1
                        or not SHA_PATTERN.fullmatch(row.get("pre_feedback_model_visible_messages_sha256", ""))
                        or type(row.get("first_verified_payment_observed")) is not bool):
                    raise ReleaseError("No-upfront-cue request-audit row is incomplete")
                required_hashes = set(REQUIRED_SOURCE_FILES) | {"MANIFEST.json"}
                if any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("run_id"), str)
                    or not item["run_id"].startswith(row["case"] + "--")
                    or not isinstance(item.get("source_sha256"), dict)
                    or set(item["source_sha256"]) != required_hashes
                    for item in selected
                ):
                    raise ReleaseError("No-upfront-cue selected source hash inventory is incomplete")
                if any(
                    not isinstance(item, dict)
                    or item.get("source") not in STUDY_SOURCES[study_id]
                    or not isinstance(item.get("run_id"), str)
                    or not item["run_id"].startswith(row["case"] + "--")
                    or not isinstance(item.get("source_sha256"), dict)
                    or not item["source_sha256"]
                    or not set(item["source_sha256"]) <= required_hashes
                    for item in excluded
                ):
                    raise ReleaseError("No-upfront-cue excluded attempt hash inventory is incomplete")
            conversation = _inside(folder_path, relative)
            content = conversation.read_bytes()
            if _sha(content) != row["yaml_sha256"]:
                raise ReleaseError(f"New-study YAML SHA-256 differs: {conversation}")
            parsed = _check_conversation(conversation, content)
            if (study_id == "company_code_no_upfront_cue_v1"
                    and row["pre_feedback_request_count"] > sum("model_output" in item for item in parsed)):
                raise ReleaseError("No-upfront-cue audited requests exceed recorded model outputs")
            if any(next(iter(item)) not in NEW_CONVERSATION_KEYS for item in parsed):
                raise ReleaseError(f"New-study YAML contains metadata-heavy keys: {conversation}")
            if any("reasoning_summary" in item and OPAQUE_TERRA_REASONING.fullmatch(item["reasoning_summary"])
                   for item in parsed):
                raise ReleaseError(f"Opaque provider reasoning entered readable YAML: {conversation}")
            omitted = row.get("opaque_reasoning_omitted_count")
            if type(omitted) is not int or omitted < 0 or (row["arm"] != "terra" and omitted):
                raise ReleaseError(f"New-study opaque-reasoning count is malformed: {study_id}")
            omitted_counts.append(omitted)
            validated_paths.add(conversation.resolve())
            expected_links.add((relative, row["yaml_sha256"]))
            for segment in row.get("segments", []):
                if (not SHA_PATTERN.fullmatch(segment.get("plan_sha256", ""))
                        or not SHA_PATTERN.fullmatch(segment.get("freeze_receipt_sha256", ""))):
                    raise ReleaseError(f"New-study frozen-plan binding is missing: {study_id}")
                amendment = segment.get("amendment_scenario")
                if amendment is not None:
                    amendment_path = _inside(ROOT, amendment)
                    if _sha(amendment_path.read_bytes()) != segment.get("amendment_scenario_sha256"):
                        raise ReleaseError(f"New-study amendment hash differs: {study_id}")
            for segment in [*row.get("segments", []), *row.get("excluded_attempts", [])]:
                if (segment.get("source") not in EVIDENCE_ROOTS
                        or not RUN_PATTERN.fullmatch(segment.get("run_id", ""))
                        or not all(SHA_PATTERN.fullmatch(v) for v in segment.get("source_sha256", {}).values())):
                    raise ReleaseError(f"New-study source hash ledger is malformed: {study_id}")
        if table_links != expected_links:
            raise ReleaseError(f"New-study README table differs from manifest: {study_id}")
        if (study_id == "company_code_no_upfront_cue_v1"
                and manifest["pre_feedback_request_audit"]["request_count"] != sum(
                    row["pre_feedback_request_count"] for row in rows)):
            raise ReleaseError("No-upfront-cue audited request totals differ")
        omitted_total = sum(omitted_counts)
        if (manifest.get("opaque_reasoning_omitted_count") != omitted_total
                or manifest.get("terra_reasoning_availability") != (
                    "encrypted_provider_content_omitted_no_readable_summary_recorded"
                    if omitted_total else "as_recorded")
                or (omitted_total > 0 and
                    f"{omitted_total} opaque Terra provider reasoning payload(s)" not in readme_text)):
            raise ReleaseError(f"New-study reasoning-availability note differs: {study_id}")
        actual = {p.resolve() for p in folder_path.rglob("*.yaml")}
        if actual != {p for p in validated_paths if p.is_relative_to(folder_path.resolve())}:
            raise ReleaseError(f"New-study folder contains unlisted YAML: {study_id}")
        if study_id == "company_code_no_upfront_cue_v1":
            expected_files = {readme, *(folder_path / row["conversation"] for row in rows)}
            expected_dirs = {folder_path / "conversations"}
            expected_dirs.update(folder_path / "conversations" / arm for arm in ARMS)
            entries = set(folder_path.rglob("*"))
            if (entries != expected_files | expected_dirs
                    or any(entry.is_symlink() for entry in entries)):
                raise ReleaseError("No-upfront-cue public folder contains an unlisted file or directory")
        _privacy_check(readme, readme.read_bytes())
        _privacy_check(path, path.read_bytes())
        studies[study_id] = {"folder": folder, "rows": len(rows)}
    return {"new_studies": studies, "public_yaml_count": len(validated_paths),
            "privacy_check": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="Validate an explicit selection, write nothing")
    modes.add_argument("--publish", action="store_true", help="Publish a validated new study once")
    modes.add_argument("--verify-public", action="store_true", help="Verify new public studies without private evidence")
    parser.add_argument("--selection", type=Path, help="Explicit private selection JSON for dry run or publication")
    parser.add_argument("--local-root", type=Path, default=ROOT / ".local")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.verify_public:
        print(json.dumps(verify_public(args.results_root.resolve()), sort_keys=True))
        return 0
    if args.selection is None:
        raise ReleaseError("--selection is required for dry run or publication")
    if args.dry_run:
        _, manifest = prepare(args.selection.resolve(), args.local_root.resolve())
        print(json.dumps({"study_id": manifest["study_id"], "rows": manifest["row_count"],
                          "folder": manifest["folder"], "privacy_check": "passed", "written": False}, sort_keys=True))
        return 0
    print(json.dumps(publish(args.selection.resolve(), args.local_root.resolve(), args.results_root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
