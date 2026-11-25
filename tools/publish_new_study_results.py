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
from tools.publish_readable_results import (  # noqa: E402
    ReleaseError, _check_conversation, _dump_conversation, _inside,
    _privacy_check, _read_json, _read_jsonl, _sha, _timestamp,
)

SELECTION_SCHEMA = "research4.readable-new-study-selection.v1"
MANIFEST_SCHEMA = "research4.readable-new-study-release.v1"
FOLDER_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}Z--[a-z0-9-]+$")
RUN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARMS = ("sp_lense_rank16_all64", "qwen38_27b_base", "terra", "deepseek_flash")

SCENARIOS = {
    "infrastructure_split_gpu_layout_v3": Path("scenarios/infrastructure_split_gpu_layout_v3/cases.yaml"),
    "company_code_access_v1": Path("scenarios/company_code_access_v1/cases.yaml"),
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
}
DRIVER_ROOTS = {
    "infrastructure_split_gpu_layout_v3": Path("2026-11-23_10--split-gpu-layout-v3-run"),
    "company_code_access_v1": Path("2026-11-23_10--company-code-access-v1-run"),
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
    cells = {(arm, case) for arm in arms for case in cases}
    if (len(cases) != len(set(cases)) or len(cells) != scenario["design"]["total_planned_episodes"]):
        raise ReleaseError(f"Frozen scenario has duplicate or missing cells: {study_id}")
    return cells


def _source_dir(local_root: Path, arm: str, segment: dict) -> Path:
    source, run_id = segment.get("source"), segment.get("run_id")
    if source not in EVIDENCE_ROOTS or not isinstance(run_id, str) or not RUN_PATTERN.fullmatch(run_id):
        raise ReleaseError("Unknown evidence source or unsafe run ID")
    relative, has_arm = EVIDENCE_ROOTS[source]
    path = local_root / relative / arm / run_id if has_arm else local_root / relative / run_id
    if not path.resolve().is_relative_to(local_root.resolve()):
        raise ReleaseError("Evidence source escapes the private root")
    if not path.is_dir():
        raise ReleaseError(f"Missing selected evidence: {source}/{arm}/{run_id}")
    return path


def _check_attempt_inventory(local_root: Path, study_id: str, arm: str,
                             case: str, run_prefix: str, row: dict) -> None:
    """Require every existing attempt, including setup-only directories, to be accounted for."""
    observed = set()
    for source in STUDY_SOURCES[study_id]:
        if (source == "company_suffix" and arm != "sp_lense_rank16_all64"
                or source == "split_terra_suffix" and (arm, case) != ("terra", "split_w2_n1")):
            continue
        relative, has_arm = EVIDENCE_ROOTS[source]
        source_dir = local_root / relative / arm if has_arm else local_root / relative
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
                          expected_cells: set[tuple[str, str]], scenario: dict) -> list[dict]:
    """Account for structured setup failures without publishing driver details."""
    driver_root = local_root / DRIVER_ROOTS[study_id]
    observed_receipts = {path.name: _sha(path.read_bytes())
                         for path in driver_root.glob("*DRIVER*.json") if path.is_file()}
    if selection.get("driver_receipts_sha256") != observed_receipts:
        raise ReleaseError(f"Driver receipt inventory or hashes differ: {study_id}")
    source_rows = selection.get("precall_dispositions")
    if not isinstance(source_rows, list):
        raise ReleaseError(f"Setup disposition list is missing: {study_id}")
    condition_to_case = ({item["condition_id"]: item["case_id"] for item in scenario["cases"]}
                         if study_id == "infrastructure_split_gpu_new_seed_supplement_v1" else {})
    observed = Counter()
    for filename in observed_receipts:
        receipt = _read_json(driver_root / filename)
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


def _protocol_binding(local_root: Path, study_id: str, scenario: dict,
                      scenario_sha: str, arm: str, case: str, segment: dict,
                      parent: dict | None) -> dict[str, str]:
    """Bind a selected raw segment to its frozen plan and amendment receipt."""
    source = segment["source"]
    run_dir = _source_dir(local_root, arm, segment)
    manifest_path = run_dir / "MANIFEST.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
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
    return content, {
        "arm": arm, "case": case, "status": expected_status,
        "outcome_valid": expected_status == "complete",
        "finished_utc": last_result["finished_utc"],
        "first_segment_finished_utc": finish_times[0],
        "elapsed_hours": last_result["score"].get("elapsed_hours"),
        "termination_reason": last_result["score"].get("termination_reason"),
        "segments": source_rows, "excluded_attempts": excluded,
    }


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
    condition_by_case = {item["case_id"]: item["condition_id"]
                         for item in scenario["cases"]} if study_id == "infrastructure_split_gpu_new_seed_supplement_v1" else {}
    rows = selection.get("rows")
    expected = _expected_cells(study_id, scenario)
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or {(r.get("arm"), r.get("case")) for r in rows} != expected):
        raise ReleaseError(f"Selection does not cover the complete frozen roster: {study_id}")
    precall = _precall_dispositions(selection, local_root, study_id, expected, scenario)
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
        _check_attempt_inventory(local_root, study_id, arm, case, run_prefix, row)
        if study_id == "company_code_access_v1":
            expected_sources = ("company_primary", "company_suffix") if arm == "sp_lense_rank16_all64" else ("company_primary",)
            if sources != expected_sources:
                raise ReleaseError(f"Company prefix/suffix selection differs: {arm}/{case}")
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
    if earliest.strftime("%Y-%m-%d_%HZ") != folder[:14]:
        raise ReleaseError("Cohort folder hour differs from earliest included source result")
    manifest_rows.sort(key=lambda r: (ARMS.index(r["arm"]), r["case"]))
    manifest = {
        "schema": MANIFEST_SCHEMA, "study_id": study_id,
        "scenario": scenario_path.relative_to(ROOT).as_posix(),
        "scenario_sha256": scenario_sha,
        "selection_sha256": _sha(selection_path.read_bytes()),
        "folder": folder, "row_count": len(manifest_rows),
        "precall_count": sum(row["count"] for row in precall),
        "precall_dispositions": precall,
        "rows": manifest_rows,
    }
    title = study_id.replace("_", " ").capitalize()
    manifest_link = "../../" + _manifest_path(study_id).as_posix()
    lines = [f"# {title}", "", "Readable selected Game conversations. Model-visible text is copied exactly from recorded requests and responses. Source IDs, dispositions and SHA-256 hashes are in the [versioned catalog manifest](" + manifest_link + "); raw evidence is not in this release.", "", f"Scenario: [{scenario_path.name}](../../{scenario_path.relative_to(ROOT).as_posix()}) (SHA-256 `{scenario_sha}`).", "", "A prefix and its separately amended suffix form one case-level observation. An incomplete or malformed attempt is shown as such and is never assigned a zero outcome. Post-hoc supplements remain separate from the primary study.", "", "| Arm | Case | Status | Simulated hours | Conversation | YAML SHA-256 |", "| --- | --- | --- | ---: | --- | --- |"]
    lines[6:6] = [STUDY_NOTES[study_id], ""]
    for r in manifest_rows:
        hours = r["elapsed_hours"] if r["elapsed_hours"] is not None else "—"
        lines.append(f"| `{r['arm']}` | `{r['case']}` | {r['status']} | {hours} | [YAML]({r['conversation']}) | `{r['yaml_sha256']}` |")
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
        folder_path = results_root / folder
        readme = folder_path / "README.md"
        if not readme.is_file():
            raise ReleaseError(f"New-study README is missing: {study_id}")
        root_readme = results_root / "README.md"
        if f"({folder}/README.md)" not in root_readme.read_text(encoding="utf-8"):
            raise ReleaseError(f"Results collection does not link new study: {study_id}")
        readme_text = readme.read_text(encoding="utf-8")
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
        for row in rows:
            relative = row.get("conversation")
            if (row.get("arm") not in ARMS or row.get("status") not in {"complete", "incomplete"}
                    or row.get("outcome_valid") is not (row["status"] == "complete")
                    or not isinstance(relative, str) or not relative.startswith(f"conversations/{row['arm']}/")
                    or not SHA_PATTERN.fullmatch(row.get("yaml_sha256", ""))):
                raise ReleaseError(f"New-study row is malformed: {study_id}")
            conversation = _inside(folder_path, relative)
            content = conversation.read_bytes()
            if _sha(content) != row["yaml_sha256"]:
                raise ReleaseError(f"New-study YAML SHA-256 differs: {conversation}")
            parsed = _check_conversation(conversation, content)
            if any(next(iter(item)) not in NEW_CONVERSATION_KEYS for item in parsed):
                raise ReleaseError(f"New-study YAML contains metadata-heavy keys: {conversation}")
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
        actual = {p.resolve() for p in folder_path.rglob("*.yaml")}
        if actual != {p for p in validated_paths if p.is_relative_to(folder_path.resolve())}:
            raise ReleaseError(f"New-study folder contains unlisted YAML: {study_id}")
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
