"""One source-locked integration batch, no retries and no real backend release."""
import copy
import json
import os
import sys
import time
from support import HERE,FIXTURES,require,sha,write_new,bounds


def pure_boundaries():
    from authority import check_chain,authenticate,independent_execution
    from shared_cleanup import SharedCleanup
    raws=[(HERE/n).read_bytes() for n in ("SOURCE_FREEZE.json","RELEASE.json","AUTHORIZATION.json")]
    lock=json.loads((HERE/"AUTHORITY_LOCK.json").read_bytes())
    expected=os.environ["SP_CONFIRMATION_AUTHORITY_SHA"]
    check_chain(*raws,lock,expected)
    rejected=[]
    for name,change in (("source_byte",lambda r,l:r.__setitem__(0,r[0]+b" ")),
        ("release_byte",lambda r,l:r.__setitem__(1,r[1]+b" ")),
        ("authorization_byte",lambda r,l:r.__setitem__(2,r[2]+b" ")),
        ("wrong_lock",lambda r,l:l.__setitem__("release_sha256","0"*64))):
        r,l=list(raws),copy.deepcopy(lock)
        change(r,l)
        try: check_chain(*r,l,expected)
        except ValueError: rejected.append(name)
        else: raise AssertionError("forged chain accepted")
    execution=authenticate()["execution"]
    independent_execution(execution)
    for field,value in (("mode","REAL_QWEN"),("production_authorized",True),("real_model_work",True)):
        bad={**execution,field:value}
        try: independent_execution(bad)
        except ValueError: rejected.append("mode_"+field)
        else: raise AssertionError("injection relabeled by reader")
    # Prospectively allow a different properly linked authorization without
    # changing the source hash: demonstrates acyclicity, not permission to run.
    alternate=json.loads(raws[2]);alternate["fixtures"]=list(reversed(alternate["fixtures"]))
    alternate_raw=(json.dumps(alternate,sort_keys=True,indent=2)+"\n").encode()
    other={**lock,"authorization_sha256":sha(alternate_raw)}
    other_sha=sha((json.dumps(other,sort_keys=True,indent=2)+"\n").encode())
    check_chain(raws[0],raws[1],alternate_raw,other,other_sha)
    shared=SharedCleanup(100.)
    first=shared.deadlines(100.,"worker")
    require(first==(1900.,1915.),"worker exact 1800 plus shared15")
    shared.charge("worker",6.)
    second=shared.deadlines(1906.,"audit")
    require(second==(2086.,2095.) and shared.remaining==9.,"audit cannot mint another15 seconds")
    shared.charge("audit",9.)
    try: shared.deadlines(2095.,"audit")
    except ValueError: rejected.append("exhausted_cleanup")
    else: raise AssertionError("cleanup renewed")
    over=SharedCleanup(0.)
    over.charge("worker",8.)
    try: over.charge("audit",7.000001)
    except ValueError: rejected.append("cleanup_plus_one")
    else: raise AssertionError("overrun accepted")
    require(over.used>15.,"overrun retained, never capped in receipt")
    return {"rejections":rejected,"acyclic_alternate_authorization_same_sources":True,"shared_cleanup":shared.record(),
        "overrun_true_seconds":over.used,"production_authorized":False}


def main():
    require(len(sys.argv)==2,"caller supplies exact authority-lock SHA")
    os.environ["SP_CONFIRMATION_AUTHORITY_SHA"]=sys.argv[1]
    usage_raw=(HERE/"USAGE_RECEIPT.json").read_bytes()
    usage=json.loads(usage_raw)
    require(0<=usage["derived"]["used_percent"]<100 and 0<=time.time()-usage["observed_unix_seconds"]<=120,"fresh available standard usage before batch")
    started=time.monotonic();cutoff=started+570.;absolute=started+600.
    write_new("BATCH_STARTED.json",{"observed_unix_seconds":time.time(),"started_monotonic":started,
        "substantive_cutoff":cutoff,"absolute_end":absolute,"usage_sha256":sha(usage_raw),"attempt":1},preparation=True)
    expected=json.loads((HERE/"EXPECTATIONS.json").read_bytes())
    report={"schema":"confirmation_integration_batch.v2","status":"INCONCLUSIVE_INTEGRATION_ONLY",
        "groups":[{"name":name,"status":"UNRUN"} for name in expected["order"]],
        "real_model_loads":0,"real_forwards":0,"tokenizer_loads":0,"production_authorized":False,"failure":None}
    active=None
    try:
        from production_run import controller,control_file
        for active in report["groups"]:
            require(time.monotonic()<cutoff,"immutable outer substantive cutoff")
            active["status"]="STARTED"
            if active["name"]=="boundaries":
                active["details"]=pure_boundaries()
            else:
                name=active["name"]
                os.environ["SP_CONFIRMATION_FIXTURE"]=name
                final=controller(cutoff)
                audit=json.loads(control_file("AUDIT_RESULT.json").read_bytes())
                worker=json.loads(control_file("WORKER_RESULT.json").read_bytes())
                want=expected["fixtures"][name]
                active["parent_final"]=final
                active["independent_audit"]=audit
                require(final["audit_completed"] is True and final["classification"]==want["classification"]
                    and final["worker_quiescent"] and final["audit_quiescent"] and not final["controller_errors"],"valid complete external controller/audit outcome")
                for key in ("classification","forward_attempts","derivatives","loads","routes","self_endpoints","strict_flips","retentions","off_identities","cell_counts","request_counts"):
                    require(audit[key]==want[key],"prospectively exact synthetic count: "+key)
                terminal=worker["terminal"]
                status=terminal["recorder_status"]
                require(terminal["execution"]["mode"]=="INJECTION_MODULE" and terminal["execution"]["real_model_work"] is False,
                    "never relabel injected module as Qwen")
                require(audit["hook_result"]["hook_status"]==want["hook_status"],"complete normal versus typed negative versus technical hook status")
                require(terminal["runtime_record"]["hook_checks"]==want["hook_checks"]
                    and status.get("terminal_cleanup_checks",0)==want["terminal_checks"],"normal/terminal hook counts remain distinct")
                if name=="hook_io":
                    root=control_file(".").parent/"evidence"/"attempt"
                    require(status["receipt_failed"] is True and status["terminal"] is True and not terminal["complete_evidence"]
                        and not (root/"hook_evidence/fault.json").exists() and (root/"hook_evidence/checks.jsonl").stat().st_size==7,
                        "missing fault receipt and partial append survive outer closeout")
                elif "scientific_kind" in want:
                    require([x["kind"] for x in audit["scientific_failures"]]==[want["scientific_kind"]]
                        and audit["hook_result"]["prefix_verified"] is True and audit["hook_result"]["permits_pass"] is False,
                        "valid negative evidence is not scientific PASS")
                else: require(audit["hook_result"]["permits_pass"] is True,"full109 synthetic evidence")
            active["status"]="PASS"
            print(json.dumps({"group":active["name"],"status":"PASS","elapsed_seconds":time.monotonic()-started}),flush=True)
        require(time.monotonic()<=absolute,"absolute600 includes closeout")
        report["status"]="PASS_INTEGRATION_ONLY"
    except BaseException as error:
        if active is not None: active["status"]="FAIL"
        import traceback
        report["failure"]={"code":"FIXED_INTEGRATION_GROUP_FAILURE","type":type(error).__name__,
            "frames":[{"file":frame.filename.split("\\")[-1].split("/")[-1],"line":frame.lineno,"function":frame.name}
                for frame in traceback.extract_tb(error.__traceback__)[-8:]]}
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,finished_monotonic=time.monotonic(),
            substantive_cutoff=cutoff,absolute_end=absolute,within_absolute=time.monotonic()<=absolute,
            bounds=bounds(),source_freeze_sha256=sha((HERE/"SOURCE_FREEZE.json").read_bytes()),
            authority_lock_sha256=sys.argv[1],remaining_unrun=sum(g["status"]=="UNRUN" for g in report["groups"]))
        write_new("BATCH_REPORT.json",report,critical=True,preparation=True)
    return 0 if report["status"]=="PASS_INTEGRATION_ONLY" else 1


if __name__=="__main__":raise SystemExit(main())
