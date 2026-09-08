"""One saved-FAKE-evidence batch at actual optimization 0/1/2; no parameters."""
import collections
import copy
import hashlib
import importlib.abc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from support import HERE,ROOT,SOURCES,require,sha,bounds,check_freeze,write_new
from real_boundary import encoded,strict,write_exclusive,usage_value

class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".",1)[0] in {"torch","transformers","transformer_lens","tokenizers","sp_lense","numpy","safetensors","huggingface_hub"}:
            raise RuntimeError("FORBIDDEN_RESEARCH_IMPORT")
        return None

def save(relative,value,raw=False):
    data=value if raw else encoded(value)
    require(len(data)<=5*1024**2,"unchanged per-file cap")
    write_exclusive(HERE/relative,data,HERE)
    bounds()
    return {"path":relative,"bytes":len(data),"sha256":sha(data)}

def lock_check(approved):
    raw=(HERE/"BATCH_LOCK.json").read_bytes();require(sha(raw)==approved,"caller-pinned batch lock")
    lock=strict(raw)
    for name,key in (("SOURCE_FREEZE.json","source_sha256"),("TEST_PROTOCOL.json","test_protocol_sha256"),("TEST_INPUTS.json","test_inputs_sha256")):
        require(sha((HERE/name).read_bytes())==lock[key],"frozen source/case/input bytes")
    check_freeze()
    return lock

def fixture_pair(kind):
    pins=strict((HERE/"FIXTURE_PINS.json").read_bytes())["files"]
    stem="test_evidence/"+kind+"/real_evidence/fresh_confirmation_weight_order_fix_attempt_001/control/"
    values=[]
    for name in ("LOADER_DIAGNOSTICS.json","FIX_TERMINAL.json"):
        pin=pins[stem+name];raw=SOURCES.read(pin["commit"],pin["path"])
        require(len(raw)==pin["bytes"] and sha(raw)==pin["sha256"],"immutable saved FAKE fixture bytes")
        values.append(strict(raw))
    require(values[1]["model_work"]=="BYTE_STANDINS_ONLY","never reinterpret real evidence")
    require(values[1]["diagnostic"]["receipt_sha256"]==sha(SOURCES.read(pins[stem+"LOADER_DIAGNOSTICS.json"]["commit"],pins[stem+"LOADER_DIAGNOSTICS.json"]["path"])),"original native terminal receipt")
    return values[0],values[1]["diagnostic"]

def transform(case,level,source_sha):
    native,terminal=fixture_pair(case["source_fixture"])
    old_execution=copy.deepcopy(native["execution"])
    execution={"schema":"synthetic_saved_reader_execution.v1","mode":"SAVED_FAKE_TRANSFORMATION_ONLY",
        "attempt_id":"fresh_confirmation_weight_order_fix_attempt_002","case":case["name"],"optimization_level":level,
        "production_authorized":False,"original_fake_execution_sha256":sha(encoded(old_execution))}
    native["execution"]=copy.deepcopy(execution);terminal["execution"]=copy.deepcopy(execution)
    proof=native["legacy_weight_binding"];proof["execution"]=copy.deepcopy(execution);proof["source_sha256"]=source_sha
    name=case["name"]
    if name=="source":proof["source_sha256"]="0"*64
    if name=="execution":proof["execution"]["synthetic_inner_tamper"]=True
    if name=="phase":proof["phases"][1]="FORGED_PHASE"
    if name=="multiplicity":
        current=proof["current_after"];current[0]["identity"]=current[1]["identity"];seen=collections.Counter()
        for row in current:
            seen[row["identity"]]+=1;row["occurrence"]=seen[row["identity"]]
    if name=="byte_boundary":proof["current_after"][0]["byte_length"]+=4
    if name=="digest":
        proof["constructor_sha256"]="0"*64;native["already_computed_digests"]["actual_ordered_sha256"]="0"*64
    # Repair outer receipt and duplicated proof so inner conditions are actually reached.
    terminal["legacy_weight_binding"]=copy.deepcopy(proof)
    terminal["already_computed_digests"]=copy.deepcopy(native["already_computed_digests"])
    raw=(json.dumps(native,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    terminal["receipt_sha256"]="0"*64 if name=="receipt" else sha(raw)
    if name!="receipt":require(terminal["receipt_sha256"]==sha(raw),"repaired outer native hash")
    require(terminal["legacy_weight_binding"]==proof,"repaired duplicate proof")
    return raw,terminal,execution

def child(approved,level,cutoff):
    require(sys.flags.optimize==level,"ACTUAL_OPTIMIZATION_LEVEL_MISMATCH")
    sys.meta_path.insert(0,NoResearch())
    lock=lock_check(approved);protocol=strict((HERE/"TEST_PROTOCOL.json").read_bytes())
    require(level in protocol["optimization_levels"],"frozen actual optimization group")
    from source_parity import check
    parity=check()
    import weight_reader
    actual_need=weight_reader.need;failures=[]
    def observed_need(value):
        if not value:failures.append(sys._getframe(1).f_lineno)
        return actual_need(value)
    weight_reader.need=observed_need
    result={"schema":"optimized_saved_reader_group.v1","status":"INCONCLUSIVE_SAVED_READER",
        "requested_optimization":level,"observed_optimization":sys.flags.optimize,"source_sha256":lock["source_sha256"],
        "cases":[dict(name=x["name"],status="UNRUN") for x in protocol["cases"]],"source_parity":parity,
        "real_model_work":False,"parameter_accesses":0,"parameter_hash_calls":0,"constructors":0,"failure":None}
    active=None
    try:
        for spec,case in zip(protocol["cases"],result["cases"],strict=True):
            active=case;require(time.monotonic()<cutoff,"one shared substantive deadline")
            raw,terminal,execution=transform(spec,level,lock["source_sha256"])
            prefix="test_evidence/opt"+str(level)+"/"+spec["name"]+"/"
            native_pin=save(prefix+"NATIVE.json",raw,raw=True);terminal_pin=save(prefix+"TERMINAL.json",terminal)
            transform_pin=save(prefix+"TRANSFORMATION.json",{"schema":"explicit_synthetic_saved_fixture_rebinding.v1",
                "source_fixture":spec["source_fixture"],"source_pins_sha256":sha((HERE/"FIXTURE_PINS.json").read_bytes()),
                "synthetic_rebinding":True,"real_authority":False,"execution":execution,
                "outer_hash_repaired":spec["name"]!="receipt","terminal_proof_copy_repaired":True,
                "native":native_pin,"terminal":terminal_pin})
            saved_native=(HERE/native_pin["path"]).read_bytes();saved_terminal=strict((HERE/terminal_pin["path"]).read_bytes())
            require(sha(saved_native)==native_pin["sha256"] and sha((HERE/terminal_pin["path"]).read_bytes())==terminal_pin["sha256"],"exclusive saved readback")
            failures.clear()
            answer=weight_reader.interpret(saved_native,saved_terminal,execution,lock["source_sha256"],
                strict((HERE/"TEST_INPUTS.json").read_bytes())["synthetic_expected_weight_sha256"])
            require(answer["interpretation"]==spec["expected"] and answer["binding_verified"]==spec["verified"] and not answer["scientific_pass"],"fixed saved reader classification")
            observed=failures[0] if failures else None
            require(observed==spec["failed_need_line"],"intended mandatory condition actually reached")
            require(len(failures)<=1,"first failed need is fail closed")
            case.update(status="PASS",answer=answer,failed_need_line=observed,native=native_pin,terminal=terminal_pin,
                transformation=transform_pin,synthetic_saved_evidence_only=True)
            save(prefix+"RESULT.json",case)
            require(time.monotonic()<cutoff,"saved readers share one substantive cutoff")
        result["status"]="PASS_OPTIMIZED_SAVED_READER_ONLY"
    except BaseException:
        result["failure"]="FIXED_OPTIMIZATION_GROUP_FAILURE"
        if active is not None:active["status"]="FAIL"
    finally:
        result["finished_monotonic"]=time.monotonic()
        result["area"]=bounds()
        save("test_evidence/opt"+str(level)+"/GROUP_RESULT.json",result)
    print(json.dumps({"optimization":level,"status":result["status"],"cases_pass":sum(c["status"]=="PASS" for c in result["cases"])}))
    return 0 if result["status"]=="PASS_OPTIMIZED_SAVED_READER_ONLY" else 1

def main(approved):
    sys.meta_path.insert(0,NoResearch())
    lock=lock_check(approved);protocol=strict((HERE/"TEST_PROTOCOL.json").read_bytes())
    usage=strict((HERE/"USAGE_BEFORE_BATCH.json").read_bytes())
    require(0<=time.time()-usage["observed_unix_seconds"]<=120 and usage_value(usage["tool_result"])["used_percent"]<100,"fresh actual nonexhausted standard usage")
    python=Path(protocol["python_executable"])
    require(python.is_file() and sha(python.read_bytes())==protocol["python_executable_sha256"],"pinned direct interpreter, no launcher-child ambiguity")
    require(not (HERE/"root_release").exists() and not (HERE/"real_evidence").exists(),"real release absent")
    started=time.monotonic();cutoff=started+45;absolute=started+60
    timer=threading.Timer(59.5,lambda:os._exit(124));timer.daemon=True;timer.start()
    write_new("BATCH_STARTED.json",{"started":started,"substantive_cutoff":cutoff,"absolute_deadline":absolute,
        "batch_lock_sha256":approved,"source_sha256":lock["source_sha256"],"real_model_work":False},preparation=True,critical=True)
    result={"schema":"one_optimized_saved_reader_batch.v1","status":"INCONCLUSIVE_SAVED_READER_BATCH",
        "groups":[{"optimization":n,"status":"UNRUN"} for n in protocol["optimization_levels"]],
        "source_sha256":lock["source_sha256"],"batch_lock_sha256":approved,"model_loads":0,"parameter_accesses":0,
        "parameter_hash_calls":0,"constructors":0,"forwards":0,"derivatives":0,"encoding":0,"old_suites_rerun":0,
        "model_imports":0,"failure":None,"owned_model_processes":0,"scientific_pass":False}
    process=None;active=None;closed=[];shared_cleanup=0.0
    try:
        from launch import preflight
        disabled=preflight("0"*64)
        require(not disabled["production_authorized"],"actual disabled-state preflight")
        write_new("DISABLED_STATE.json",disabled,preparation=True,critical=True)
        for group in result["groups"]:
            active=group;level=group["optimization"];require(time.monotonic()<cutoff,"single batch work allowance")
            args=[str(python),"-B","-E"]+(["-O"] if level==1 else ["-OO"] if level==2 else [])+[str(HERE/"reader_tests.py"),"--child",approved,str(level),str(cutoff)]
            process=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output,_=process.communicate(timeout=max(.01,cutoff-time.monotonic()))
            require(len(output)<=65536,"bounded complete pure child capture")
            capture=save("test_evidence/opt"+str(level)+"/stdout.log",output,raw=True)
            exit_code=process.returncode
            require(exit_code is not None,"retained pure child exited")
            close_started=time.monotonic()
            if hasattr(process,"_handle"):process._handle.Close()
            if process.stdout is not None:process.stdout.close()
            shared_cleanup+=time.monotonic()-close_started
            closed.append({"optimization":level,"exit_code":exit_code,"retained_handle_closed":True,"pipes_closed":True,
                "quiescent":True,"direct_interpreter":str(python),"no_pid_rediscovery":True,"no_tree_kill":True,"capture":capture})
            process=None
            group_raw=(HERE/("test_evidence/opt"+str(level)+"/GROUP_RESULT.json")).read_bytes();saved=strict(group_raw)
            group.update(status="PASS" if exit_code==0 and saved["status"]=="PASS_OPTIMIZED_SAVED_READER_ONLY" else "FAIL",
                observed_optimization=saved["observed_optimization"],cases_pass=sum(c["status"]=="PASS" for c in saved["cases"]),
                group_result_sha256=sha(group_raw),exit_code=exit_code)
            require(group["status"]=="PASS" and saved["requested_optimization"]==saved["observed_optimization"]==level,"closed actual optimization group succeeds")
            require(time.monotonic()<cutoff,"all pure saved audits within substantive deadline")
        result["status"]="PASS_SAVED_READER_OPTIMIZATION_ONLY"
    except BaseException:
        result["failure"]="SAVED_READER_BATCH_FAILURE"
        if active is not None and active["status"]=="UNRUN":active["status"]="FAIL"
    finally:
        close_started=time.monotonic()
        cleanup_before_final=shared_cleanup
        if process is not None:
            try:
                if process.poll() is None:process.terminate()
                process.communicate(timeout=max(.01,min(15-shared_cleanup,absolute-time.monotonic()-1)))
                quiescent=process.returncode is not None
                if hasattr(process,"_handle"):process._handle.Close()
                if process.stdout is not None:process.stdout.close()
                closed.append({"optimization":active["optimization"],"exit_code":process.returncode,
                    "retained_handle_closed":True,"pipes_closed":True,"quiescent":quiescent,"forced_closeout":True})
            except BaseException:
                closed.append({"optimization":active["optimization"],"quiescent":False,"closeout_incomplete":True})
        shared_cleanup+=time.monotonic()-close_started
        result.update(elapsed_seconds=time.monotonic()-started,substantive_cutoff=cutoff,absolute_deadline=absolute,
            owned_pure_children=closed,shared_cleanup_seconds=shared_cleanup,shared_cleanup_maximum_seconds=15,
            shared_cleanup_snapshot_excludes_final_receipt_writes=True,area=bounds())
        if shared_cleanup>15 or time.monotonic()>absolute or any(not c["quiescent"] for c in closed):
            result["status"]="INCONCLUSIVE_SAVED_READER_BATCH";result["failure"]="CLOSEOUT_BOUNDARY_FAILURE"
        write_new("BATCH_REPORT.json",result,preparation=True,critical=True)
        complete_cleanup=cleanup_before_final+time.monotonic()-close_started
        write_new("BATCH_CLOSEOUT.json",{"shared_cleanup_seconds":complete_cleanup,"within_shared15":complete_cleanup<=15,
            "absolute_envelope_met":time.monotonic()<=absolute,"all_pure_children_quiescent":all(c["quiescent"] for c in closed),
            "no_owned_model_processes":True},preparation=True,critical=True)
        timer.cancel()
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="PASS_SAVED_READER_OPTIMIZATION_ONLY" else 1

if __name__=="__main__":
    if len(sys.argv)==5 and sys.argv[1]=="--child":raise SystemExit(child(sys.argv[2],int(sys.argv[3]),float(sys.argv[4])))
    require(len(sys.argv)==2,"one approved prospective batch lock")
    raise SystemExit(main(sys.argv[1]))
