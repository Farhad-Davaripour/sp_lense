"""Independent saved startup reader. Never imports a writer, engine, or backend."""
import json
import time
import zlib
from support import HERE,SOURCES,require,sha,bounds
from real_boundary import checked_path,strict

def judge(boundary,execution,deadline):
    require(time.monotonic()<deadline,"saved setup audit deadline")
    root=boundary.output;control=root/"control";evidence=root/"evidence/attempt"
    def read(path):
        path=checked_path(root,path)
        require(path.stat().st_size<=5*1024**2,"saved per-file bound")
        return path.read_bytes()
    def obj(path): return strict(read(path))
    worker_raw=read("control/WORKER_RESULT.json");worker=strict(worker_raw)
    binding=obj("control/CLOSED_WORKER_BINDING.json")
    require(binding["execution"]==execution and binding["worker_result_sha256"]==sha(worker_raw)
        and binding["owned_capture_sha256"]==sha(read("control/owned/production_worker/CAPTURE.json"))
        and binding["quiescent"] is True,"closed owned worker hashes")
    require(worker["terminal"]["execution"]==execution,"worker authority identity")
    result={"schema":"independent_saved_setup_assessment.v1","execution":execution,"audit_completed":True,
        "classification":"INCONCLUSIVE_SETUP","scientific_pass":False,"real_scientific_result":False,
        "baselines_unrun":24,"requests_unrun":48,"scientific_cells_unrun":180,"forwards":0,"derivatives":0,
        "first_failure":None,"setup_evidence_complete":False,"reader_model_imports":0,"checks":[]}
    if not (control/"SETUP_TERMINAL.json").is_file():
        result["first_failure"]="MISSING_SETUP_TERMINAL";return result
    raw=read("control/SETUP_TERMINAL.json");terminal=strict(raw)
    pointer=worker["setup_result"]["setup_terminal"]
    require(pointer=={"path":(control/"SETUP_TERMINAL.json").relative_to(HERE).as_posix(),"bytes":len(raw),"sha256":sha(raw)},"native terminal acknowledgement")
    require(terminal["execution"]==execution and terminal["scientific_pass"] is False and terminal["scientific_status"]=="UNRUN","setup is never scientific PASS")
    result["first_failure"]=terminal["first_failure"]
    from authority import current
    if boundary.mock:
        from mock_setup import mock_plan
        plan=mock_plan()
    else:
        from plan import build_plan
        plan=build_plan()
    expected_state={"execution":execution,"scope":"STARTUP_ONLY_NO_QUESTIONS","cell_status":{c["cell_id"]:"UNRUN" for c in plan["cells"]},
        "request_status":{r["request_id"]:"UNRUN" for r in plan["requests"]},"baselines_unrun":24,"scientific_status":"UNRUN"}
    require(terminal["state"]==expected_state and len(plan["cells"])==180 and len(plan["requests"])==48,"exact scientific UNRUN schedule")
    counts=terminal["counts"];result["counts"]=counts
    require(counts["reserved_load_attempts"]==1 and counts["actual_load_dispatches"]<=1 and counts["sealed"] is True
        and counts["forwards"]==counts["derivatives"]==counts["tokenizer_calls"]==0,"hard startup-only counters")
    result["checks"].append("EXACT_UNRUN_AND_HARD_COUNTERS")
    reservation=obj("control/SETUP_RESERVATION.json")
    # Reconstruct allocation independently; do not import the reservation writer.
    caps={"diagnostic":128*1024,"terminal":256*1024,"index":256*1024,"other_closeout":128*1024} if boundary.mock else {
        "diagnostic":2*1024**2,"terminal":9*1024**2//4,"index":5*1024**2,"other_closeout":3*1024**2//4}
    require(reservation["caps"]==caps and reservation["before_load"] is True and reservation["load_max"]==1
        and reservation["forward_max"]==reservation["derivative_max"]==reservation["tokenizer_max"]==0
        and len(raw)<=caps["terminal"],"reserved startup receipt caps")
    index_ref=terminal["index"]
    require(index_ref is not None,"saved accounting required")
    indexraw=read("evidence/attempt/"+index_ref["path"]);index=strict(indexraw)
    require(len(indexraw)==index_ref["bytes"]<=caps["index"] and sha(indexraw)==index_ref["sha256"],"saved index length and hash")
    require(index["schema"]=="sp_lense.confirmation_io_index.v2" and index["execution"]==execution,"index execution mode")
    contract=strict(SOURCES.read("2620f66d4d50456c88800554bc9a26845998cf50","diagnostics/semantic_confirmation_resource_v1/contract.json"))
    expected={"seconds":contract["seconds"],"storage":contract["storage"],"study":contract["study"],
        "codec":contract["logit_codec"]["name"],"row_maximum_bytes":contract["row_maximum_bytes"],"hook_metadata":contract["hook_metadata"]}
    require(index["settings"]==expected,"unchanged recorder worker audit reader contract")
    reconciliation=index["reconciliation"];seen=set();total=0
    require(reconciliation.get("workflow")==expected_state,"saved accounting schedule")
    for item in reconciliation["files"]:
        key=item["path"];require(key not in seen and key==key.casefold(),"unique canonical evidence")
        data=read("evidence/attempt/"+key);seen.add(key);total+=len(data)
        require(len(data)==item["actual_bytes"] and sha(data)==item["actual_sha256"],"actual evidence reconciliation")
        if item["complete"]:
            require(len(data)==item["expected_bytes"]==item["recorded_bytes"] and sha(data)==item["expected_sha256"]==item["recorded_sha256"],"complete IO identity")
    files={p.relative_to(evidence).as_posix() for p in evidence.rglob("*") if p.is_file()}
    require(files==seen|{index_ref["path"]} and total==reconciliation["actual_total_bytes"]
        and total+len(indexraw)==index_ref["actual_total_including_index"],"no untracked or omitted actual evidence")
    result["checks"].append("RAW_INDEX_SETTINGS_AND_RECONCILIATION")
    diagnostic=terminal["loader_diagnostics"]
    require(diagnostic is not None and diagnostic["execution"]==execution and diagnostic["permits_scientific_pass"] is False
        and diagnostic["retry_allowed"] is False,"finite diagnostic provenance")
    component_sha=sha(SOURCES.read("84bfd749876c44dc9bb1bc1e75db89f3e491c159","diagnostics/fresh_confirmation_loader_diagnostics_v1/SOURCE_FREEZE.json"))
    require(diagnostic["source_sha256"]==component_sha,"checked diagnostic source identity")
    if diagnostic["receipt_sha256"] is not None:
        diagnosticraw=read("control/LOADER_DIAGNOSTICS.json");published=strict(diagnosticraw)
        require(len(diagnosticraw)<=caps["diagnostic"] and sha(diagnosticraw)==diagnostic["receipt_sha256"],"native diagnostic exact acknowledgement")
        require(published["execution"]==execution and published["source_sha256"]==component_sha,"native diagnostic chain")
        for field in ("stage","stage_events","first_failure","parameter_enumerations","already_computed_digests"):
            require(published[field]==diagnostic[field],"first-cause and enumeration retained")
    result["diagnostic_stage"]=diagnostic["stage"]
    result["recorder_status"]=terminal["recorder_status"]
    result["guard"]=terminal["guard"]
    result["native_diagnostic_complete"]=diagnostic["complete_diagnostic_evidence"]
    if terminal["status"]!="SETUP_DIAGNOSTIC_COMPLETE":
        require(terminal["status"]=="INCONCLUSIVE_SETUP","no unknown successful status")
        bounds();return result
    require(not terminal["failures"] and terminal["first_failure"] is None and counts["stop_reason"] is None and counts["actual_load_dispatches"]==1
        and counts["load_dispatch_attempts_including_denied"]==1,"one clean startup")
    require(binding["good_capture"] is True and index["status"]=="COMPLETE" and not index["sticky_failure"] and not index["failures"]
        and not reconciliation["issues"],"clean authoritative accounting")
    require(diagnostic["stage"]=="ADAPTER_READY" and diagnostic["complete_diagnostic_evidence"] is True
        and diagnostic["diagnostic_io_failed"] is False,"complete native loader receipt")
    guard=terminal["guard"];cleanup=terminal["cleanup"]
    require(guard["forwards"]==guard["derivatives"]==guard["rejected"]==0 and guard["restored"] is True and guard["first_dispatch_code"] is None,"restored original zero-dispatch guard")
    require(cleanup["complete"] is True and cleanup["normal_scientific_hook_checks"]==0 and cleanup["full_scientific_finalizer_called"] is False,"setup cleanup only")
    parameter=cleanup["parameter_state"]
    require(all(v for v in parameter.values() if type(v) is bool),"unchanged parameter cleanup predicates")
    expected_weight=diagnostic["already_computed_digests"]["expected_frozen_sha256"]
    require(parameter["parameter_sha256"]==parameter["initial_parameter_sha256"]==expected_weight
        ==diagnostic["already_computed_digests"]["actual_ordered_sha256"],"exact original ordered digest equality")
    status=terminal["recorder_status"]
    require(status==diagnostic["recorder_status"] and status["latch_state"]=="ADMITTED" and status["terminal"] is False
        and status["primary_code"] is None and status["secondary_codes"]==[] and status["receipt_failed"] is False
        and status["index_sha256"] is None and status["permits_pass"] is False
        and status["remaining_ids"]==[c["cell_id"] for c in plan["cells"]],"untouched full scientific schedule")
    receipt=obj("evidence/attempt/hook_evidence/setup_receipt.json")
    require(receipt["complete"] is True and receipt["forward_calls"]==0 and receipt["reserved_checks"]==109
        and receipt["check_record_bytes"]==4096 and receipt["fault_reserve_bytes"]==receipt["index_reserve_bytes"]==65536,"complete original hook admission")
    source_raw=read("evidence/attempt/hook_evidence/source_lock.json");source=strict(source_raw)
    require(sha(source_raw)==receipt["source_lock_sha256"] and source["passed"] is True and source["before_materialization"] is True,"authenticated setup source receipt")
    if boundary.mock: require(source["mock_only"] is True and receipt["declared_setup_receipt"]["injected_tensor_module_only"] is True,"honest inert setup provenance")
    else:
        spec=strict((HERE/"RUNTIME_SPEC.json").read_bytes())
        require(source["installed_sources_sha256"]==spec["installed_sources_sha256"]
            and receipt["declared_setup_receipt"]["injected_tensor_module_only"] is False,"actual installed-source setup binding")
    reconstructed={}
    for name,reference in receipt["values"].items():
        require(name in ("setup_before","reference","setup_changes") and reference["manifest"]=="hook_evidence/"+name+".json","exact setup values")
        manifest=obj("evidence/attempt/"+reference["manifest"]);pieces=[]
        require(manifest["complete"] is True and manifest["raw_bytes"]<=64*1024**2,"complete bounded setup chunks")
        for ordinal,chunk in enumerate(manifest["chunks"]):
            require(chunk["path"]=="hook_evidence/%s_%03d.zlib"%(name,ordinal),"exact chunk order")
            compressed=read("evidence/attempt/"+chunk["path"])
            require(len(compressed)==chunk["compressed_bytes"] and sha(compressed)==chunk["compressed_sha256"],"raw compressed identity")
            decoder=zlib.decompressobj();piece=decoder.decompress(compressed,4*1024**2+1)
            require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and len(piece)==chunk["raw_bytes"]<=4*1024**2
                and sha(piece)==chunk["raw_sha256"],"lossless bounded complete chunk")
            pieces.append(piece)
        joined=b"".join(pieces)
        require(len(joined)==manifest["raw_bytes"]==reference["raw_bytes"] and sha(joined)==manifest["raw_sha256"]==reference["raw_sha256"],"full setup reconstruction")
        reconstructed[name]=strict(joined)
    require(set(reconstructed)=={"setup_before","reference","setup_changes"}
        and cleanup["reference_sha256"]==cleanup["current_registry_sha256"]==receipt["values"]["reference"]["raw_sha256"],"clean setup inspection matches exact admitted reference")
    require(not any(p in files for p in ("hook_evidence/index.json","hook_evidence/checks.jsonl","hook_evidence/fault.json","hook_evidence/prefix_index.json")),"no scientific hook finalization or faults")
    result["checks"].append("LOSSLESS_SETUP_AND_CLEANUP_IDENTITY")
    result.update(classification="SETUP_DIAGNOSTIC_COMPLETE",setup_evidence_complete=True)
    bounds();require(time.monotonic()<deadline,"saved audit completes within deadline")
    return result
