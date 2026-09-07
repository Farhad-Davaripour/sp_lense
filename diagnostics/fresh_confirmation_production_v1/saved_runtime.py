"""Independent real-adapter evidence joins; never loads a model or gate worker."""
import json
from support import HERE, require
from hook_binding import component


def verify_runtime_identity(root,state,io):
    require(io["status"] == "COMPLETE" and not io["errors"], "complete independently reconciled writer evidence")
    record = json.loads((root/"runtime_adapter.json").read_bytes())
    plan = json.loads((root/"plan.json").read_bytes())
    spec = json.loads((HERE/"RUNTIME_SPEC.json").read_bytes())
    compatibility = spec["runtime_compatibility"]
    require(record["fake_backend"] is False and record["input_bound"] is True, "actual measured adapter record")
    require(all(record["metadata"][k] == compatibility[k] for k in
        ("model_id","model_revision","device","dtype","d_model","model_layers","packages","lens")), "independent fitted-runtime metadata join")
    identity = record["final_identity"]
    require(identity["initial_parameter_flags"] == identity["current_parameter_flags"] and
        identity["initial_parameter_versions"] == identity["current_parameter_versions"] and
        len(identity["initial_parameter_flags"]) == len(identity["initial_parameter_versions"]) == identity["parameter_count"],
        "independent full flag/version arrays and parameter count")
    required = ("parameter_identities_unchanged","parameter_versions_unchanged","parameter_gradients_absent",
        "parameter_flags_restored","parameter_bytes_unchanged","hook_registry_restored","active_request_empty",
        "wrapper_cache_empty","bridge_cache_empty")
    require(all(identity[k] is True for k in required) and identity["parameter_sha256"] ==
        identity["initial_parameter_sha256"] == record["initial_parameter_sha256"] == compatibility["weight_sha256"], "actual parameter and cold-state identity evidence")
    labels = []
    categories = {p["prompt_id"]:p["category"] for p in plan["prompts"]}
    for r in plan["requests"]:
        labels.append("REQUEST-entry:"+r["request_id"])
        if categories[r["prompt_id"]] == "self":
            labels.append("ON-entry:"+r["request_id"])
        labels.append("REQUEST-exit:"+r["request_id"])
    labels.append("matrix-finally")
    require(len(labels) == record["hook_checks"] == 109, "independent complete normal hook schedule")
    source_lock = {"installed_sources_sha256":spec["installed_sources_sha256"],"passed":True,"before_materialization":True}
    hook = component().judge(root,labels,[c["cell_id"] for c in plan["cells"]],source_lock,
                             controller_status=record["hook_controller_status"])
    require(hook["permits_pass"] is True, "complete independently authenticated matching hook evidence")
    measured = state["adapter_accounting"]
    require(measured["forwards"] == state["attempts"]["forward"] and measured["derivatives"] == state["attempts"]["derivative"]
        and measured["rejected_dispatches"] == 0 and measured["dispatch_failed"] is False
        and state["cleanup_complete"] is True, "actual guarded calls equal schedule accounting with no terminal latch")
    return hook
