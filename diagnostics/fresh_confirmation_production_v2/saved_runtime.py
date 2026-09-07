"""Independent mode, real identity, normal/prefix hook and external status joins."""
import json
from support import HERE,require,sha
from authority import independent_execution
from hook_binding import component


def labels_for(plan):
    categories={p["prompt_id"]:p["category"] for p in plan["prompts"]}
    labels=[]
    for r in plan["requests"]:
        labels.append("REQUEST-entry:"+r["request_id"])
        if categories[r["prompt_id"]]=="self": labels.append("ON-entry:"+r["request_id"])
        labels.append("REQUEST-exit:"+r["request_id"])
    labels.append("matrix-finally")
    require(len(labels)==109,"unchanged complete hook denominator")
    return labels


def verify_runtime_identity(root,state,io,capture):
    execution=independent_execution(state["execution"])
    terminal=capture["terminal"]
    require(terminal["execution"]==execution and terminal["state"]==state,"closed external terminal/accounting equality")
    record=terminal["runtime_record"]
    if io["status"]!="COMPLETE" or io["errors"] or record is None or state["technical_failures"]:
        return {"evidence_valid":False,"permits_pass":False,"hook_status":"INCOMPLETE"}
    require(json.loads((root/"runtime_adapter.json").read_bytes())==record and
        record["execution"]==execution and record["hook_controller_status"]==terminal["recorder_status"],"ordinary saved record equals closed-controller status")
    spec=json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())
    if execution["mode"]=="REAL_QWEN":
        compatibility=spec["runtime_compatibility"]
        require(all(record["metadata"][k]==compatibility[k] for k in
            ("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")),"actual fitted-runtime identity")
        expected_digest=compatibility["weight_sha256"]
    else:
        fixture=json.loads((HERE/"INJECTION_SPEC.json").read_bytes())
        require(record["metadata"]==fixture["runtime_metadata"],"explicit injection metadata, never Qwen provenance")
        expected_digest=fixture["parameter_sha256"]
    identity=record["final_identity"]
    require(identity is not None and all(identity[k] is True for k in
        ("parameter_identities_unchanged","parameter_versions_unchanged","parameter_gradients_absent","parameter_flags_restored",
         "parameter_bytes_unchanged","hook_registry_restored","active_request_empty","wrapper_cache_empty","bridge_cache_empty")),"exact actual parameter/cold receiver invariants")
    require(identity["initial_parameter_flags"]==identity["current_parameter_flags"] and
        identity["initial_parameter_versions"]==identity["current_parameter_versions"] and
        len(identity["initial_parameter_flags"])==len(identity["initial_parameter_versions"])==identity["parameter_count"],"complete flag/version identities")
    require(identity["parameter_sha256"]==identity["initial_parameter_sha256"]==record["initial_parameter_sha256"]==expected_digest,"actual full parameter bytes")
    plan=json.loads((root/"plan.json").read_bytes())
    labels=labels_for(plan)
    source={"installed_sources_sha256":spec["installed_sources_sha256"],"passed":True,"before_materialization":True}
    schedule=[c["cell_id"] for c in plan["cells"]]
    if state["scientific_failures"]:
        finding=state["scientific_failures"]
        require(len(finding)==1,"one fail-stop scientific finding")
        raw=(json.dumps(finding[0],sort_keys=True,separators=(",",":"))+"\n").encode()
        # Prefix expectation derives from raw scientific ledger + independent
        # completed cell/request reconstruction, not the recorder's own finding.
        completed_checks=0
        for request in plan["requests"]:
            status=state["request_status"][request["request_id"]]
            category=next(p["category"] for p in plan["prompts"] if p["prompt_id"]==request["prompt_id"])
            if status=="DONE": completed_checks+=3 if category=="self" else 2
            elif status=="FAILED":
                completed_checks+=1
                # A completed gradient/endpoint proves the ON-entry preceded it.
                if any(c["request_id"]==request["request_id"] and c["phase"] not in ("baseline","entry")
                    and state["cell_status"][c["cell_id"]]!="UNRUN" for c in plan["cells"]): completed_checks+=1
                break
        expected={"reason_code":finding[0]["kind"].upper(),"finding_sha256":sha(raw),"cursor":state["cursor"],"completed_checks":completed_checks}
        hook=component().judge_prefix(root,labels,schedule,source,controller_status=terminal["recorder_status"],expected_scientific_stop=expected)
        valid=hook.get("prefix_verified") is True and hook["permits_pass"] is False
    else:
        hook=component().judge_full(root,labels,schedule,source,controller_status=terminal["recorder_status"])
        valid=hook["permits_pass"] is True
    measured=state["adapter_accounting"]
    require(measured["forwards"]==state["attempts"]["forward"] and measured["derivatives"]==state["attempts"]["derivative"]
        and measured["rejected_dispatches"]==0 and measured["dispatch_failed"] is False and state["cleanup_complete"] is True,"guarded attempts independently equal schedule")
    return {**hook,"evidence_valid":valid}
