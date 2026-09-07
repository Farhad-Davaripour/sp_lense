"""Closed exact-f03_v2_first_C production admission. Disabled during this preparation."""
import importlib.metadata,json,math,sys,time
from pathlib import Path
from core import HERE,ROOT,read,require,sha,git,check_freeze,json_bytes
from usage_receipt import derive_standard_usage,strict_json
from select_inputs import check_selection_freeze

def exact_cohort(plan):
    check_selection_freeze();freeze=check_freeze()
    require(plan["execution_mode"]=="PRODUCTION_F03_V2_FIRST_C" and plan["fixture_scope"]=="exact_census_selected_f03_v2_self_STOP_first_C_request","fake/substituted cohort rejected")
    raw=(HERE/"production_plan.json").read_bytes()
    require(sha(raw)==freeze["source_sha256"]["production_plan.json"] and plan==json.loads(raw),"exact committed production plan")
    lock=read(HERE/"input_lock.json")
    source=git("show",lock["input_commit"]+":"+HERE.relative_to(ROOT).as_posix()+"/inputs.json")
    require(source==(HERE/"inputs.json").read_bytes() and sha(source)==lock["inputs_sha256"],"independent committed selected prompt bytes")
    locked=json.loads(source);require(plan["prompts"]==locked["prompts"] and plan["requests"]==locked["requests"],"no prompt/request substitution")
    require((len(plan["prompts"]),len(plan["requests"]),len(plan["cells"]),len(plan["derivative_cells"]))==(1,1,11,4),"exact entire assay schedule")
    require(plan["self_prompt_ids"]==[p["prompt_id"] for p in plan["prompts"]] and len(plan["self_request_ids"])==1 and plan["ordinary_truths"]=={},"self-only exact denominator")
    from inputs import build_plan
    require(plan==build_plan(),"independent schedule and token-lock derivation")
    return {"prompts":1,"requests":1,"forwards":11,"derivatives":4,"fresh_routes_max":2,"endpoints_max":1,"off_checks":0}

def environment(plan):
    sources=read(HERE/"source_bindings.json");contract=plan["gate"]["runtime_compatibility"]
    require(sha(Path(sys.executable).read_bytes())==contract["environment"]["executable_sha256"],"venv executable")
    require(sha(Path(sys._base_executable).read_bytes())==sources["owned_identity"]["base_sha256"],"reviewed current base executable")
    for name,version in sources["packages"].items():require(importlib.metadata.version(name)==version,"package "+name)
    for path,digest in {**sources["external_sha256"],**plan["hook_integration"]["installed_sources_sha256"]}.items():
        require(sha(Path(path).read_bytes())==digest,"external runtime/helper/tokenizer "+path)
    cache=read(HERE/"model_cache_lock.json")
    for name,row in cache["files"].items():
        path=Path(cache["snapshot"])/name;require(path.stat().st_size==row["bytes"] and sha(path.read_bytes())==row["sha256"],"pinned model cache bytes")
    require(sha((HERE/"fitted_parameters.json").read_bytes())==plan["gate"]["parameter_sha256"],"unchanged frozen gate")
    return True

def validate_release(value,raw,expected_sha,plan,now):
    require(expected_sha is not None and sha(raw)==expected_sha,"root release must be separately pinned, not caller asserted")
    require(strict_json(raw)==value,"declared release object differs from pinned raw document")
    require(value["authorizer"]=="root" and value["scope"]=="ONE_CENSUS_SELECTED_F03_V2_FIRST_C_ASSAY","exact root release scope")
    source_freeze=git("show",value["source_commit"]+":"+HERE.relative_to(ROOT).as_posix()+"/freeze.json")
    require(source_freeze==(HERE/"freeze.json").read_bytes(),"release binds the earlier committed frozen source, not a circular release commit hash")
    require(value["freeze_sha256"]==sha((HERE/"freeze.json").read_bytes()) and value["input_lock_sha256"]==sha((HERE/"input_lock.json").read_bytes()),"release source/input binding")
    require(value["limits"]=={"forwards":11,"derivatives":4,"loads":1,"worker_seconds":300,"cleanup_seconds":15,"audit_seconds":90,"bytes":96*1024**2},"no alternate limits")
    usage=value["usage"]
    require(set(usage)=={"source","bucket","captured_at_unix","used_percent","tool_receipt"},"ambiguous/missing usage declaration fields")
    timestamp=usage["captured_at_unix"]
    require(type(now) in (int,float) and math.isfinite(now) and type(timestamp) in (int,float) and math.isfinite(timestamp),"finite receipt timestamps")
    require(usage["source"]=="get_usage_limits" and usage["bucket"]=="standard_codex" and 0<=now-timestamp<=120,"fresh authenticated usage")
    require(type(usage["used_percent"]) in (int,float) and math.isfinite(usage["used_percent"]) and 0<=usage["used_percent"]<100,"known unexhausted standard usage")
    require(value["usage_receipt_sha256"]==sha(json_bytes(usage["tool_receipt"])),"fresh tool receipt bound by root release")
    actual=derive_standard_usage(usage["tool_receipt"])
    require(usage["used_percent"]==actual["used_percent"],"declared percentage differs from independently derived highest standard window")
    require(actual["used_percent"]<100,"actual standard usage exhausted")
    return True


def release_documents():
    policy=read(HERE/"authorization.json")
    require(policy["run_authorized"] is True and policy["root_release_sha256"] is not None,"NO REAL MODEL RELEASE: root pin required")
    prefix=HERE.relative_to(ROOT).as_posix()
    require(git("show","HEAD:"+prefix+"/authorization.json")==(HERE/"authorization.json").read_bytes(),"committed root authorization")
    raw=(HERE/"approved_root_release.json").read_bytes()
    require(git("show","HEAD:"+prefix+"/approved_root_release.json")==raw and sha(raw)==policy["root_release_sha256"],"committed exact root release")
    return policy,raw,strict_json(raw)

def admit_production():
    policy,raw,value=release_documents();plan=read(HERE/"production_plan.json")
    exact_cohort(plan);environment(plan)
    validate_release(value,raw,policy["root_release_sha256"],plan,time.time())
    return plan,value

def admit_audit():
    # Launch freshness naturally expires; audit authenticates the identical release
    # actually admitted at launch, sources and environment, not a fabricated new timestamp.
    policy,raw,value=release_documents();plan=read(HERE/"production_plan.json")
    exact_cohort(plan);environment(plan)
    saved=read(HERE/"real_attempt/release.json")
    require(saved["root_release"]==value and saved["root_release_sha256"]==policy["root_release_sha256"],"same launched authority for saved audit")
    require(value["freeze_sha256"]==sha((HERE/"freeze.json").read_bytes()) and value["input_lock_sha256"]==sha((HERE/"input_lock.json").read_bytes()),"unchanged launched pins")
    return plan,value
