"""Closed production admission and independent exact-cohort input authentication."""
import importlib.metadata
import json,math,sys,time
from pathlib import Path
from core import HERE,ROOT,read,require,sha,git,check_freeze,json_bytes
from usage_receipt import derive_standard_usage,strict_json

INPUT_COMMIT="1438e20d6c235de2c25dd01c95cc1764599bdf17"
INPUT_NS="diagnostics/semantic_editor_final_inputs_v1"
INPUT_INVENTORY="2a5108b7ea861407edc335329a6f193da5115b32ac96bfb50e7551a68a47b293"
INPUT_LOCK="0744ea76b5c3711f8139e9194e5e2276307f09c71c99cd9c9ed5c702aa6e6d59"
INPUTS_SHA="bd91aa64b97c77e5e7de2c0ba995b598fd69c78c3c03518f997689ccd9b74591"
def archived(path):return git("show",INPUT_COMMIT+":"+INPUT_NS+"/"+path)

def exact_cohort(plan):
    """Judge-side admission compares raw frozen source, not runner self-consistency."""
    require(plan["execution_mode"]=="PRODUCTION_FINAL" and plan["fixture_scope"]=="exact_fixed_final_cohort","synthetic/compact evidence is not production")
    frozen=check_freeze();raw=(HERE/"production_plan.json").read_bytes()
    require(sha(raw)==frozen["source_sha256"]["production_plan.json"] and plan==json.loads(raw),"exact frozen full production plan")
    require(sha(archived("FINAL_INVENTORY.json"))==INPUT_INVENTORY,"independent source inventory")
    inv={r["path"]:r for r in json.loads(archived("FINAL_INVENTORY.json"))["files"]}
    source=archived("inputs.json");require(sha(source)==INPUTS_SHA==inv["inputs.json"]["sha256"],"independent exact prompt source")
    locked=json.loads(source);lock_raw=archived("input_lock.json");require(sha(lock_raw)==INPUT_LOCK,"independent input lock")
    require(len(plan["prompts"])==24 and len(plan["requests"])==48 and len(plan["cells"])==180 and len(plan["derivative_cells"])==48,"exact conservative schedule")
    require(len(plan["self_prompt_ids"])==6 and len(plan["self_request_ids"])==12,"exactself denominators")
    require([{k:v for k,v in r.items() if k!="sign"} for r in plan["requests"]]==locked["requests"],"exact policy schedule")
    for p,original,binding in zip(plan["prompts"],locked["prompts"],json.loads(lock_raw)["prompts"],strict=True):
        require(all(p[k]==v for k,v in original.items()),"no substituted prompt/metadata")
        proofraw=archived(binding["tokens"]["path"])
        require(sha(proofraw)==binding["tokens"]["sha256"]==inv[binding["tokens"]["path"]]["sha256"],"independent token proof source")
        proof=json.loads(proofraw);require(plan["alignment"][p["prompt_id"]]==proof,"complete frozen IDs/masks/roles")
    return {"prompts":24,"requests":48,"forwards":180,"derivatives":48,"fresh_routes":72,"self_endpoints":12,"off_checks":36,"max_input_length":136}

def environment(plan):
    contract=plan["gate"]["runtime_compatibility"]
    require(sha(Path(sys.executable).read_bytes())==contract["environment"]["executable_sha256"],"frozen Python executable")
    for name,version in contract["packages"].items():require(importlib.metadata.version(name)==version,"frozen package "+name)
    require(sha((ROOT/plan["model"]["config_path"]).read_bytes())==plan["model"]["config_sha256"],"model config identity")
    for path,digest in plan["hook_integration"]["installed_sources_sha256"].items():require(sha(Path(path).read_bytes())==digest,"installed hook source "+path)
    for path,digest in read(HERE/"runtime_source_receipt.json")["external_sha256"].items():require(sha(Path(path).read_bytes())==digest,"external runtime source "+path)
    return True

def validate_release(value,raw,expected_sha,plan,now):
    require(expected_sha is not None and sha(raw)==expected_sha,"root release must be separately pinned, not caller asserted")
    require(strict_json(raw)==value,"declared release object differs from pinned raw document")
    require(value["authorizer"]=="root" and value["scope"]=="ONE_REAL_FINAL_ASSESSMENT","exact root release scope")
    source_freeze=git("show",value["source_commit"]+":"+HERE.relative_to(ROOT).as_posix()+"/freeze.json")
    require(source_freeze==(HERE/"freeze.json").read_bytes(),"release binds the earlier committed frozen source, not a circular release commit hash")
    require(value["freeze_sha256"]==sha((HERE/"freeze.json").read_bytes()) and value["input_lock_sha256"]==INPUT_LOCK,"release source/input binding")
    require(value["limits"]=={"forwards":180,"derivatives":48,"loads":1,"worker_seconds":1500,"cleanup_seconds":15,"audit_seconds":180,"bytes":288*1024**2},"no alternate limits")
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

def admit_production():
    policy=read(HERE/"authorization.json")
    require(policy["run_authorized"] is True and policy["root_release_sha256"] is not None,"NO REAL MODEL RELEASE: root review required")
    prefix=HERE.relative_to(ROOT).as_posix()
    require(git("show","HEAD:"+prefix+"/authorization.json")== (HERE/"authorization.json").read_bytes(),"root release pin must be committed separately")
    plan=read(HERE/"production_plan.json");exact_cohort(plan);environment(plan)
    raw=(HERE/"approved_root_release.json").read_bytes();value=strict_json(raw)
    require(git("show","HEAD:"+prefix+"/approved_root_release.json")==raw,"root-approved exact release artifact must be committed")
    validate_release(value,raw,policy["root_release_sha256"],plan,time.time())
    return plan,value
