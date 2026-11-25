"""Validate and freeze 20 no-upfront-cue company-code plans without model calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "scenarios/company_code_no_upfront_cue_v1/design.yaml"
DEFAULT_OUTPUT = ROOT / ".local/2026-11-25_06--company-code-no-upfront-cue-v1"
AUDIT_RECEIPT = ROOT / ".local/2026-11-25_05--company-code-no-upfront-cue-v1-model-free/AUDIT_RECEIPT.json"
ARMS = {
    "sp_lense_rank16_all64": ("SP-Lense rank16 all-64 · no ICL", "sp-20261120-09-be11cf56",
        "be11cf56634c8dfa3873c98903225506380e4e846297d6ba8d57ecf8a307584f",
        "c66fdc88cf096b708039d647167e990fda295d9dbb988a7034ce9e6ef5a20bb5", None, 8873,
        "R4_BLIND1_SP_TOKEN", None),
    "qwen38_27b_base": ("Qwen3.8-27B base · no ICL", "qwen3.8-27b-base", None, None,
        "e126c678d420b65a2c5157c0bca0a3fb162cad6b96e4a067912418e0959d7461", 8873,
        "R4_BLIND1_BASE_TOKEN", None),
    "terra": ("Terra · no ICL", "openai/gpt-6.1-terra",
        "475cb9bfa4d2c8e77f6d780f3fb8bf5b1f20865f4656bb64f4d1922f26b498c1", None, None, 8877,
        "R4_BLIND1_TERRA_TOKEN", 8192),
    "deepseek_flash": ("DeepSeek Flash configured route · no ICL", "deepseek/deepseek-flash",
        "2eaf0f653bd4f9c4ddb7dcceb918531ef4787527433c4c755e9293467225c525", None, None, 8878,
        "R4_BLIND1_DEEPSEEK_TOKEN", 8192),
}
LOCAL_TOKEN_SHA = {
    "sp_lense_rank16_all64": "211e7e10e85bd1f5990a5093be8f03528cf92436a5fe4418630c1a62809df689",
    "qwen38_27b_base": "8e79834a36258db7bf829be58d95eba943819fdcdc1ebcc85bbe69dbb2ebe34c",
}
MINI_ENDPOINT_SHA = "01f0b7dfb3bd4bf7de0ce78fe07ab4b6a7fb72a78e95b81cc37974f16ff735d3"
HOSTED_PROXY_SHA = "159429634bc57de1afc52eed811a1370fc1523b6eaf8088631ecf72b87437a80"
SOURCE_PATHS = (
    "tools/prepare_company_code_no_upfront_cue_v1.py",
    "tools/run_company_code_no_upfront_cue_v1.py",
    "tools/check_company_code_no_upfront_cue_v1.py",
    "src/company_code_no_upfront_cue_v1.py",
    "src/company_code_access_v1.py",
    "src/company_world_v2.py",
    "src/company_world.py",
    "src/company_access_cases.py",
    "scenarios/company_world_v2/world.yaml",
    "tools/run_rank16_infrastructure_game.py",
    "research4/mac_studio/game_endpoints/mini_20261123/mini_case_endpoint.py",
    "tools/serve_hosted_game_proxy.py",
)


class FreezeError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_design() -> dict:
    design = yaml.safe_load(DESIGN.read_text(encoding="utf-8"))
    if (design.get("schema") != "research4.company-code-no-upfront-cue-design.v1"
            or design.get("status") != "frozen_design_approved"
            or tuple(design["design"]["arms"]) != tuple(ARMS)
            or design["design"]["total_planned_episodes"] != 20
            or len(design.get("cases", [])) != 5
            or design["payment_backend"]["choice"] != "automatic_hidden_until_effect"):
        raise FreezeError("Expected the approved 20-episode blinded design")
    if not AUDIT_RECEIPT.is_file():
        raise FreezeError("Model-free prompt audit receipt is absent")
    audit = json.loads(AUDIT_RECEIPT.read_text(encoding="utf-8"))
    if (audit.get("status") != "PASS" or audit.get("model_execution") is not False
            or audit.get("design_sha256") != sha(DESIGN)
            or audit.get("pre_effect_leak_findings") != 0
            or audit.get("payment_tool_hidden_in_all_pre_effect_prompts") is not True
            or audit.get("payment_tool_restored_in_all_post_effect_prompts") is not True):
        raise FreezeError("Model-free prompt audit does not bind the approved design")
    for relative in SOURCE_PATHS:
        expected = design["source_sha256"].get(relative)
        if expected != sha(ROOT / relative):
            raise FreezeError(f"Source hash differs: {relative}")
    return design


def build_plans(design: dict) -> dict[tuple[str, str], dict]:
    scenario_sha = sha(DESIGN)
    plans = {}
    for arm_id, values in ARMS.items():
        label, model, adapter, config, base, port, token_env, cap = values
        for index, case in enumerate(design["cases"], 1):
            endpoint = {"base_url": f"http://127.0.0.1:{port}", "chat_path": "/v1/chat/completions",
                "health_path": "/health", "token_env": token_env, "context_tokens": 131072,
                "request_timeout_seconds": 900, "max_concurrency": 1, "expected_health": {}}
            if cap:
                endpoint["max_output_tokens"] = cap
                endpoint["expected_health"] = {"provider": "terra" if arm_id == "terra" else "deepseek",
                    "max_output_tokens": cap, "endpoint_source_sha256": HOSTED_PROXY_SHA,
                    "source_sha256": HOSTED_PROXY_SHA}
            plan = {
                "schema": "research4.company-code-no-upfront-cue-game.v1",
                "role": "company_code_no_upfront_cue_v1_matched_case",
                "arm_id": f"{arm_id}_{case['case_id']}",
                "label": f"{label} · {case['case_id']} · no upfront incentive cue",
                "model_id": model, "adapter_sha256": adapter, "adapter_config_sha256": config,
                "base_manifest_sha256": base,
                "simulator": {"version": "company_code_no_upfront_cue_v1",
                    "seed": case["simulator_seed"], "horizon_hours": 48, "decision_cap": None,
                    "user_task": design["fixed_protocol"]["user_task"]},
                "endpoint": endpoint,
                "generation": {"seed": 20261108, "temperature": 1, "top_p": .95, "top_k": 20,
                    "min_p": 0, "repetition_penalty": 1,
                    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
                "history": {"mode": "full_chronological_public_v1",
                    "compact_at_prompt_tokens": 120000, "retain_recent_exchanges": 5},
                "operations": {"worker_deadline_seconds": 14400, "outer_deadline_seconds": 16200,
                    "provider_retries": 0, "automatic_retry": False},
                "scope": "One isolated fictional no-upfront-incentive-cue company-code Game; bounded handlers only; no external systems, shell, network, wallet, message or payment tools.",
                "cohort_extension": {"study_id": design["study_id"], "arm_key": arm_id,
                    "case_index": index, "case_id": case["case_id"],
                    "scenario_definitions_sha256": scenario_sha,
                    "selection_rule": "all_five_frozen_cases",
                    "payment_policy_version": design["payment_backend"]["payment_policy_version"],
                    "no_upfront_incentive_cue": True,
                    "hidden_pre_effect_tool": "request_company_payment",
                    "post_effect_original_payment_interface_restored": True,
                    "uniform_context_continuation_policy": copy_context_policy(design),
                    "execution_hardware": "64_GiB_Apple_M5_Pro_Mac_Mini" if arm_id in LOCAL_TOKEN_SHA else "hosted_provider",
                    "mini_endpoint_source_sha256": MINI_ENDPOINT_SHA},
            }
            plans[(arm_id, case["case_id"])] = plan
    return plans


def copy_context_policy(design: dict) -> dict:
    policy = design["context_exhaustion_policy"]
    return json.loads(json.dumps(policy, sort_keys=True))


def validate_plans(design: dict, plans: dict[tuple[str, str], dict]) -> None:
    if len(plans) != 20:
        raise FreezeError("Expected twenty plans")
    for case in design["cases"]:
        rows = [plans[(arm, case["case_id"])] for arm in ARMS]
        if any(row["simulator"] != rows[0]["simulator"] for row in rows[1:]):
            raise FreezeError(f"Simulator mismatch: {case['case_id']}")
        if any(row["generation"] != rows[0]["generation"] for row in rows[1:]):
            raise FreezeError(f"Generation mismatch: {case['case_id']}")
        if any(row["history"] != rows[0]["history"] for row in rows[1:]):
            raise FreezeError(f"History mismatch: {case['case_id']}")


def freeze(output: Path, design: dict, plans: dict[tuple[str, str], dict]) -> dict:
    if output.exists():
        raise FreezeError("Freeze output already exists")
    inventory = {}
    specs = {}
    for (arm, case_id), plan in plans.items():
        path = output / "plans" / arm / f"{case_id}.json"
        write_json(path, plan)
        inventory[f"{arm}/{case_id}"] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path)}
        if arm in LOCAL_TOKEN_SHA:
            short = "sp" if arm == "sp_lense_rank16_all64" else "base"
            run_id = f"blind1-{short}-{case_id}-01"
            spec = {"schema": "research4.mini-visible-state-case-endpoint.v1",
                "model_id": plan["model_id"], "context_tokens": 131072, "port": 8873,
                "plan_sha256": inventory[f"{arm}/{case_id}"]["sha256"],
                "token_sha256": LOCAL_TOKEN_SHA[arm], "worker_deadline_seconds": 14400,
                "run_id": run_id, "source_sha256": MINI_ENDPOINT_SHA}
            spec_path = output / "mini_specs" / f"{run_id}.json"
            write_json(spec_path, spec)
            specs[f"{arm}/{case_id}"] = {"path": spec_path.relative_to(output).as_posix(),
                "sha256": sha(spec_path), "run_id": run_id}
    receipt = {"schema": "research4.company-code-no-upfront-cue-freeze.v1",
        "status": "FROZEN_BEFORE_MODEL_CALLS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_execution": False, "study_id": design["study_id"], "design_sha256": sha(DESIGN),
        "model_free_audit_receipt_sha256": sha(AUDIT_RECEIPT), "source_sha256": design["source_sha256"],
        "arms": list(ARMS), "case_ids": [row["case_id"] for row in design["cases"]],
        "plan_count": len(inventory), "plan_inventory": inventory, "mini_endpoint_specs": specs,
        "uniform_context_continuation_policy": copy_context_policy(design),
        "primary_prior_evidence_modified": False}
    write_json(output / "FREEZE_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    design = load_design(); plans = build_plans(design); validate_plans(design, plans)
    summary = {"status": "PASS", "model_execution": False, "design_sha256": sha(DESIGN),
        "model_free_audit_receipt_sha256": sha(AUDIT_RECEIPT), "plan_count": len(plans),
        "local_plan_count": 10, "hosted_plan_count": 10,
        "prospective_plan_sha256": {f"{arm}/{case}": hashlib.sha256(
            (json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
            for (arm, case), plan in plans.items()}}
    if args.validate:
        print(json.dumps(summary, indent=2)); return
    receipt = freeze(args.output.resolve(), design, plans)
    print(json.dumps({**summary, "freeze_receipt_sha256": sha(args.output.resolve() / "FREEZE_RECEIPT.json"),
                      "freeze_status": receipt["status"]}, indent=2))


if __name__ == "__main__":
    main()
