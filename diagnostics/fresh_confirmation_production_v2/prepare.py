"""Model-free source, fixed synthetic expectations and input assembly."""
import ast
import difflib
import json
import math
import struct
from support import HERE,SOURCES,require,sha,write_new,bounds


def main():
    from plan import build_plan
    from bind_production import adapted_sources,V1,BASE
    from owned_production import source
    from hook_binding import component
    plan=build_plan()
    engine,judge=adapted_sources()
    for p in HERE.glob("*.py"): ast.parse(p.read_text())
    for raw in (engine,judge,source()): ast.parse(raw)
    hook=component()
    lock=json.loads((HERE/"HOOK_BINDING.json").read_bytes())
    receipt=json.loads(SOURCES.read(lock["commit"],"diagnostics/fresh_confirmation_hook_prefix_v2/TEST_RECEIPT.json"))
    require(receipt["status"]=="PASS" and len(receipt["groups"])==3 and all(g["status"]=="PASS" for g in receipt["groups"])
        and receipt["elapsed_seconds"]<=60 and receipt["model_calls"]==receipt["tokenizer_calls"]==0
        and receipt["source_freeze_sha256"]==lock["source_freeze_sha256"],"committed complete checked prefix dependency")
    fixture=json.loads(SOURCES.read("2e11335f1036532b6f79fb2927dd9616c8a09915","diagnostics/fresh_confirmation_runtime_v1/SYNTHETIC_FIXTURE.json"))
    records=fixture["records"]
    require(len(records)==24 and all(r["input_ids"]==p["input_ids"] and r["attention_mask"]==p["attention_mask"]
        and r["input_int64_le_sha256"]==p["input_int64_le_sha256"] for r,p in zip(records,plan["prompts"],strict=True)),"unchanged exact synthetic tensor table")
    params=json.loads(SOURCES.read("1d4cc39eba5fb7f987649a05f71247061f005d02","diagnostics/semantic_editor_f03_v2_first_C_v2/fitted_parameters.json"))["parameters"]
    f32=lambda x:struct.unpack("<f",struct.pack("<f",x))[0]
    first=records[0]
    state=[f32(f32(a)+f32(10*b)) for a,b in zip(params["grand_mean"],params["direction"],strict=True)]
    hn=math.sqrt(math.fsum(x*x for x in state))
    require(first["keep_intercept"]==-.1 and hn>4.,"normal first P flip has sufficient frozen step geometry")
    require(.2*hn*.0001<.04,"four bounded ON-stop updates cannot reach requested margin")
    parameter_sha=sha(struct.pack("<1025f",*([1.]*1024+[0.])))
    injection={"schema":"confirmation_tensor_injection.v2","execution_mode":"INJECTION_MODULE","real_model_predictions":False,
        "records":records,"first_self_tensor_sha256":first["input_int64_le_sha256"],"parameter_sha256":parameter_sha,
        "runtime_metadata":{"schema":"injected_tensor_runtime.v2","model_id":"TINY_INJECTION_MODULE","device":"cpu","dtype":"float32",
            "d_model":1024,"vocabulary":248320,"Qwen_weights_loaded":False,"tokenizer_loaded":False},
        "prospective_geometry":{"first_h0_norm":hn,"normal_gradient":1.,"on_stop_gradient":.0001,
            "on_stop_maximum_margin_change_bound":.2*hn*.0001}}
    write_new("INJECTION_SPEC.json",injection,preparation=True)
    write_new("BOUND_PLAN.json",plan,preparation=True)
    expectations={"schema":"fixed_integration_expectations.v2","order":["normal","baseline_stop","on_stop","hook_io","boundaries"],
        "combined_maximum_forwards":155,"combined_maximum_derivatives":10,"real_model_loads":0,"tokenizer_loads":0,
        "fixtures":{
            "normal":{"classification":"PASS_STUDY","forward_attempts":96,"derivatives":6,"loads":1,"routes":72,
                "self_endpoints":12,"strict_flips":6,"retentions":6,"off_identities":36,"cell_counts":{"DONE":96,"SKIPPED":84},
                "request_counts":{"DONE":48},"hook_checks":109,"terminal_checks":0,"hook_status":"COMPLETE"},
            "baseline_stop":{"classification":"FAIL_STUDY","forward_attempts":1,"derivatives":0,"loads":1,"routes":1,
                "self_endpoints":0,"strict_flips":0,"retentions":0,"off_identities":0,"cell_counts":{"DONE":1,"UNRUN":179},
                "request_counts":{"UNRUN":48},"hook_checks":0,"terminal_checks":1,"hook_status":"SCIENTIFIC_STOP_COMPLETE_PREFIX","scientific_kind":"finite_eligibility"},
            "on_stop":{"classification":"FAIL_STUDY","forward_attempts":34,"derivatives":4,"loads":1,"routes":25,
                "self_endpoints":1,"strict_flips":0,"retentions":0,"off_identities":0,"cell_counts":{"DONE":34,"UNRUN":146},
                "request_counts":{"FAILED":1,"UNRUN":47},"hook_checks":2,"terminal_checks":2,"hook_status":"SCIENTIFIC_STOP_COMPLETE_PREFIX","scientific_kind":"endpoint_behavior"},
            "hook_io":{"classification":"INCONCLUSIVE_STUDY","forward_attempts":24,"derivatives":0,"loads":1,"routes":24,
                "self_endpoints":0,"strict_flips":0,"retentions":0,"off_identities":0,"cell_counts":{"DONE":24,"UNRUN":156},
                "request_counts":{"UNRUN":48},"hook_checks":0,"terminal_checks":0,"hook_status":"INCOMPLETE"}},
        "production_seconds":{"worker":1800,"cleanup":15,"audit":180},"batch_seconds":{"substantive_cutoff":570,"cleanup_closeout":30,"absolute":600},
        "shared_evidence_bytes":288*1024**2,"preparation_bytes":32*1024**2,"per_file_bytes":5*1024**2,
        "no_retries":True,"production_authorized":False}
    write_new("EXPECTATIONS.json",expectations,preparation=True)
    old=SOURCES.load("prepare_v1_delta",V1,BASE+"bind_production.py")
    old_engine,old_judge=old.adapted_sources()
    delta="".join(difflib.unified_diff(old_engine.splitlines(True),engine.splitlines(True),"production_v1_engine","production_v2_engine"))
    delta+="".join(difflib.unified_diff(old_judge.splitlines(True),judge.splitlines(True),"production_v1_judge","production_v2_judge"))
    write_new("EXACT_ENGINE_DELTA.patch",delta.encode(),raw=True,preparation=True)
    predicates=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in ("norm","f32","quality","accepted","eligible")}
    require(predicates(judge)==predicates(old_judge),"numerical scientific predicates unchanged")
    write_new("PREPARATION_RECEIPT.json",{"status":"PREPARED_MODEL_FREE","engine_sha256":sha(engine.encode()),"judge_sha256":sha(judge.encode()),
        "owned_source_sha256":sha(source().encode()),"scientific_predicates_unchanged":True,"exact_prompts":24,"cold_requests":48,
        "fixed_slots":180,"metadata_capacity":"ACTUAL_SETUP_ADMISSION_REQUIRED; ARBITRARY_FAULT_OVERFLOW_TERMINAL",
        "timing":"ALLOWANCE_NOT_MEASURED_WORST_CASE","real_model_calls":0,"tokenizer_calls":0,"production_authorized":False},preparation=True)
    print(json.dumps({"status":"PREPARED_MODEL_FREE","bounds":bounds()}))


if __name__=="__main__":main()
