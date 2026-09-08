"""Saved-only diagnostic judgment; full scientific acceptance is never inferred."""
import json
import time
import zlib
from support import HERE,SOURCES,require,sha,bounds
from real_boundary import checked_path,strict
from diagnostic_support import exact_input,allowlist,CELL_ID
from trace_reader import interpret

def judge(boundary,execution,deadline):
    require(time.monotonic()<deadline,"saved diagnostic audit deadline")
    root=boundary.output;control=root/"control";evidence=root/"evidence/attempt"
    def read(name):
        p=checked_path(root,name);require(p.stat().st_size<=5*1024**2,"saved per-file cap");return p.read_bytes()
    def obj(name):return strict(read(name))
    workerraw=read("control/WORKER_RESULT.json");worker=strict(workerraw);closed=obj("control/CLOSED_WORKER_BINDING.json")
    require(closed["execution"]==execution==worker["terminal"]["execution"] and closed["worker_result_sha256"]==sha(workerraw)
        and closed["owned_capture_sha256"]==sha(read("control/owned/production_worker/CAPTURE.json")) and closed["quiescent"] is True,"closed owned execution hashes")
    result={"schema":"independent_first_forward_diagnostic.v1","execution":execution,"audit_completed":True,
        "classification":"INCONCLUSIVE_DIAGNOSTIC","scientific_pass":False,"scientific_study_unrun":True,
        "trace_verified":False,"trace_interpretation":"TRACE_INCOMPLETE_UNKNOWN","requests_unrun":48,"checks":[]}
    if not (control/"DIAGNOSTIC_TERMINAL.json").is_file():
        result["technical_failure"]="MISSING_DIAGNOSTIC_TERMINAL";return result
    raw=read("control/DIAGNOSTIC_TERMINAL.json");terminal=strict(raw)
    require(worker["diagnostic_result"]["diagnostic_terminal"]=={"path":(control/"DIAGNOSTIC_TERMINAL.json").relative_to(HERE).as_posix(),
        "bytes":len(raw),"sha256":sha(raw)},"native diagnostic terminal acknowledgement")
    require(terminal["execution"]==execution and terminal["schema"]=="first_forward_diagnostic_terminal.v1" and terminal["scientific_pass"] is False,"finite diagnostic identity")
    state=terminal["state"];counts=terminal["counts"]
    from plan import build_plan
    plan=build_plan();locked,cell=exact_input(plan)
    require(state["execution"]==execution and state["scope"]=="ONE_LOCKED_UNEDITED_BASELINE_ONLY"
        and state["scientific_status"]=="UNRUN" and state["diagnostic_cell_id"]==CELL_ID
        and state["scoring_calls"]==state["routes"]==0,"diagnostic scope not scientific scoring")
    require(set(state["cell_status"])=={x["cell_id"] for x in plan["cells"]}
        and all(v=="UNRUN" for k,v in state["cell_status"].items() if k!=CELL_ID)
        and state["cell_status"][CELL_ID] in ("UNRUN","STARTED","FAILED","DIAGNOSTIC_CAPTURED")
        and state["request_status"]=={r["request_id"]:"UNRUN" for r in plan["requests"]},"exact one-baseline prefix and full UNRUN suffix")
    require(counts["attempts"]["load"]==1 and 0<=counts["actual_load_dispatches"]<=1
        and 0<=counts["attempts"]["forward"]<=1 and counts["guarded_forwards"]<=counts["attempts"]["forward"]
        and counts["attempts"]["derivative"]==counts["derivatives"]==counts["tokenizer_calls"]==0
        and counts["sealed"] is True and counts["exact_cell_id"]==CELL_ID,"immutable 1load/max1F/0D/0encoding counter")
    result.update(counts=counts,diagnostic_cell_status=state["cell_status"][CELL_ID],forward_returned=state["forward_returned"],
        logits_published=state["logits_published"],guard_restored=terminal["guard_restored"],cleanup=terminal["cleanup"],failures=terminal["failures"])
    reservation=obj("control/DIAGNOSTIC_RESERVATION.json")
    caps={"diagnostic":2*1024**2,"terminal":9*1024**2//4,"index":5*1024**2,"other_closeout":3*1024**2//4,"trace_receipt":65536}
    require(reservation["caps"]==caps and reservation["trace_inside_other_closeout"] is True and reservation["before_load"] is True
        and reservation["fixed_cell_id"]==CELL_ID and reservation["input_lock_sha256"]==execution["input_lock_sha256"]
        and reservation["load_max"]==reservation["forward_max"]==1 and reservation["derivative_max"]==reservation["tokenizer_max"]==0
        and len(raw)<=caps["terminal"],"reserved native receipt and terminal bounds")
    proof={"binding_verified":False,"interpretation":"BINDING_INCOMPLETE","scientific_pass":False}
    if terminal["loader_diagnostics"] is not None:
        from loader_handoff import verify_loader_binding
        proof=verify_loader_binding(evidence,execution,terminal["loader_diagnostics"])
    result["loader_binding"]=proof
    from helper_reader import verify_helper
    helper_proof=verify_helper(control,execution,terminal.get("helper_status"),worker.get("helper_status"))
    result["helper_binding"]=helper_proof
    if (control/"FIRST_FORWARD_TRACE.json").is_file():
        source_files={v["source"]:v["sha256"] for v in allowlist().values()}
        trace=interpret(read("control/FIRST_FORWARD_TRACE.json"),terminal["trace_status"],execution,sha((HERE/"SOURCE_FREEZE.json").read_bytes()),source_files)
        result["trace"]=trace;result["trace_verified"]=trace["trace_verified"];result["trace_interpretation"]=trace["interpretation"]
    ref=terminal["index"]
    if ref is None or not ref.get("path"):
        result["technical_failure"]="MISSING_ACCOUNTING_INDEX";return result
    iraw=read("evidence/attempt/"+ref["path"]);index=strict(iraw)
    require(len(iraw)==ref["bytes"]<=caps["index"] and sha(iraw)==ref["sha256"] and index["execution"]==execution
        and index["schema"]=="sp_lense.confirmation_io_index.v2","same acknowledged actual index")
    contract=strict(SOURCES.read("2620f66d4d50456c88800554bc9a26845998cf50","diagnostics/semantic_confirmation_resource_v1/contract.json"))
    settings={"seconds":contract["seconds"],"storage":contract["storage"],"study":contract["study"],"codec":contract["logit_codec"]["name"],
        "row_maximum_bytes":contract["row_maximum_bytes"],"hook_metadata":contract["hook_metadata"]}
    require(index["settings"]==settings and index["reconciliation"]["workflow"]==state,"unchanged resource contract and exact diagnostic accounting")
    seen=set();total=0
    for item in index["reconciliation"]["files"]:
        key=item["path"];require(key not in seen and key==key.casefold(),"canonical unique indexed file")
        data=read("evidence/attempt/"+key);seen.add(key);total+=len(data)
        require(len(data)==item["actual_bytes"] and sha(data)==item["actual_sha256"],"complete actual-file accounting")
        if item["complete"]:require(len(data)==item["expected_bytes"]==item["recorded_bytes"]
            and sha(data)==item["expected_sha256"]==item["recorded_sha256"],"complete evidence identity")
    files={p.relative_to(evidence).as_posix() for p in evidence.rglob('*') if p.is_file()}
    require(files==seen|{ref["path"]} and total==index["reconciliation"]["actual_total_bytes"]
        and total+len(iraw)==ref["actual_total_including_index"],"no hidden external/partial evidence")
    require(not any(x.startswith(('rows/','off_returns','routing_events','requests.jsonl','derivative_events')) for x in files),"no scoring routing derivatives or continuation")
    logits={x for x in files if x.startswith('logits/')};require(logits<= {'logits/001.f32'},"only the first baseline logits artifact")
    if state["logits_published"]:require(logits=={'logits/001.f32'} and len(read('evidence/attempt/logits/001.f32'))==248320*4,"complete first full-vocabulary raw evidence")
    result.update(io_status=index["status"],actual_indexed_bytes=total+len(iraw),recorder_status=terminal["recorder_status"])
    clean=(terminal["status"]=="FIRST_FORWARD_DIAGNOSTIC_CAPTURED" and not terminal["failures"] and result["trace_verified"]
        and result["trace"].get("first_failure") is None and proof["binding_verified"] and closed["good_capture"] is True
        and state["cell_status"][CELL_ID]=="DIAGNOSTIC_CAPTURED" and state["forward_returned"] is True and state["logits_published"] is True
        and counts["actual_load_dispatches"]==counts["attempts"]["forward"]==counts["guarded_forwards"]==1
        and counts["rejected_dispatches"]==0 and counts["stop_reason"] is None and terminal["guard_restored"] is True
        and index["status"]=="COMPLETE" and not index["sticky_failure"] and not index["failures"] and not index["reconciliation"]["issues"])
    clean=clean and helper_proof["binding_verified"]
    if clean:
        cleanup=terminal["cleanup"];require(cleanup and cleanup["diagnostic_cleanup_complete"] is True and cleanup["scientific_pass"] is False
            and cleanup["latch_admitted_before_provider"] is True and cleanup["normal_scientific_hook_checks"]==0
            and cleanup["normal_hook_checks_unrun"]==109 and terminal["normal_hook_checks_unrun"]==109
            and cleanup["full_scientific_finalizer_called"] is False and terminal["full_scientific_finalizer_called"] is False,"diagnostic cleanup only")
        parameter=cleanup["parameter_state"];expected=json.loads((HERE/'RUNTIME_SPEC.json').read_bytes())["runtime_compatibility"]["weight_sha256"]
        require(all(parameter[k] is True for k in ('active_request_empty','wrapper_cache_empty','bridge_cache_empty','hook_registry_restored')),"explicit original cold-state checks")
        require(all(v for v in parameter.values() if type(v) is bool) and parameter["parameter_sha256"]==parameter["initial_parameter_sha256"]==expected,"unchanged final weight/current registry predicates")
        receipt=obj('evidence/attempt/hook_evidence/setup_receipt.json');reference=receipt["values"]["reference"]["raw_sha256"]
        require(cleanup["current_registry_sha256"]==cleanup["reference_sha256"]==reference and receipt["complete"] is True
            and receipt["reserved_checks"]==109,"exact admitted reference and unchanged hook capacity")
        status=terminal["recorder_status"]
        require(status["terminal"] is False and status["primary_code"] is None and status["index_sha256"] is None and status["permits_pass"] is False
            and status["remaining_ids"]==[c["cell_id"] for c in plan["cells"]][1:],"only first cell consumed; never full109 scientific finalizer")
        for name,reference in receipt["values"].items():
            require(name in ('setup_before','reference','setup_changes'),"fixed complete setup value")
            manifest=obj('evidence/attempt/'+reference["manifest"]);pieces=[]
            for ordinal,chunk in enumerate(manifest["chunks"]):
                require(chunk["path"]=='hook_evidence/%s_%03d.zlib'%(name,ordinal),"fixed setup chunk order")
                compressed=read('evidence/attempt/'+chunk["path"]);require(len(compressed)==chunk["compressed_bytes"] and sha(compressed)==chunk["compressed_sha256"],"compressed setup bytes")
                dec=zlib.decompressobj();piece=dec.decompress(compressed,4*1024**2+1)
                require(dec.eof and not dec.unused_data and not dec.unconsumed_tail and len(piece)==chunk["raw_bytes"]<=4*1024**2 and sha(piece)==chunk["raw_sha256"],"bounded lossless setup chunk")
                pieces.append(piece)
            joined=b''.join(pieces);require(manifest["complete"] is True and len(joined)==manifest["raw_bytes"]==reference["raw_bytes"] and sha(joined)==manifest["raw_sha256"]==reference["raw_sha256"],"complete setup manifest")
        result["classification"]="FIRST_FORWARD_DIAGNOSTIC_CAPTURED"
    bounds();require(time.monotonic()<deadline,"saved audit cutoff")
    return result
